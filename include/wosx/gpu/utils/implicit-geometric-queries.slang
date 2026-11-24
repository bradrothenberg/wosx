/*
 * ntop_aero addition to WoSX (local): reflecting (Neumann / Robin) walls defined by an implicit instead of
 * triangles. The body is a field (a conservative signed distance, > 0 in the flow and < 0 in the body) that is
 * read from 3D textures. Every wall query that Walk on Stars needs comes from evaluations of that field:
 *
 *   distance            the field value (mapped through the Kelvin inversion when the walk runs in inverted space)
 *   field value         trilinear interpolation of the grid in full float precision next to the wall. The filtered
 *                       texture lookup of the GPU has 8 bits of position precision per cell; it is used only away
 *                       from the wall, where a bound is enough
 *   ray hit             sphere tracing on the field, then bisection on its sign
 *   wall normal         gradient of the field
 *   wall sample         a point on the tangent disc at the closest wall point, moved onto the wall along the normal
 *   star radius         the largest of a doubling set of radii for which probes on the wall still face the walker
 *                       (and, with the Kelvin transform, for which the Robin reflectance stays within its bounds)
 *
 * No triangle, edge or BVH is used. 3D only.
 */

implementing wosx;
__include core.geometric_queries;
__include utils.sdf_grid_geometric_queries;
__include variance_reduction.boundary_samplers;

public interface IImplicitBody
{
    // conservative signed distance to the wall: > 0 in the flow, < 0 in the body; exact zero set
    float value(float3 x);

    // the same value with the gradient of the interpolant (cheap; used to find the wall)
    float valueAndGradient(float3 x, out float3 g);

    // smooth gradient of value(x) (central differences; used for wall normals)
    float3 gradient(float3 x);

    // true if the walk runs in the space inverted about the origin (Kelvin transform)
    bool usesKelvin();

    // the body lies inside the ball of this radius about the origin
    float boundingRadius();
};

// a scalar field at the cell centres of a regular grid (same resources as SdfGrid3D)
public struct ImplicitGrid3D
{
    private Texture3D<float> data;
    private SamplerState sampler;
    private float3 origin;
    private float3 extent;
    private float3 cellSpacing;

    public float3 getOrigin()
    {
        return origin;
    }

    public float3 getExtent()
    {
        return extent;
    }

    public float3 getCellSpacing()
    {
        return cellSpacing;
    }

    // lookup filtered by the GPU: fast, but the filter weights have 8 bits, so the result is a staircase
    // with steps of 1/256 of a cell
    public float fast(float3 x)
    {
        float3 localCoords = (x - origin) / extent;
        return data.SampleLevel(sampler, localCoords.zyx, 0);
    }

    // the stored value of one cell: at the centre of a texel the filter returns the texel itself
    private float texel(float3 i, float3 n)
    {
        float3 uvw = (i + 0.5005) / n;
        return data.SampleLevel(sampler, uvw.zyx, 0);
    }

    // trilinear interpolation in full float precision, with the gradient of the interpolant
    public float precise(float3 x, out float3 g)
    {
        float3 n = round(extent / cellSpacing);
        float3 uRaw = (x - origin) / cellSpacing - 0.5;
        float3 u = clamp(uRaw, float3(0.0), n - 1.0);
        float3 b = min(floor(u), n - 2.0);
        float3 f = u - b;

        float c000 = texel(b, n);
        float c100 = texel(b + float3(1.0, 0.0, 0.0), n);
        float c010 = texel(b + float3(0.0, 1.0, 0.0), n);
        float c110 = texel(b + float3(1.0, 1.0, 0.0), n);
        float c001 = texel(b + float3(0.0, 0.0, 1.0), n);
        float c101 = texel(b + float3(1.0, 0.0, 1.0), n);
        float c011 = texel(b + float3(0.0, 1.0, 1.0), n);
        float c111 = texel(b + float3(1.0, 1.0, 1.0), n);

        float c00 = lerp(c000, c100, f.x);
        float c10 = lerp(c010, c110, f.x);
        float c01 = lerp(c001, c101, f.x);
        float c11 = lerp(c011, c111, f.x);
        float c0 = lerp(c00, c10, f.y);
        float c1 = lerp(c01, c11, f.y);

        g.x = lerp(lerp(c100 - c000, c110 - c010, f.y), lerp(c101 - c001, c111 - c011, f.y), f.z) / cellSpacing.x;
        g.y = lerp(c10 - c00, c11 - c01, f.z) / cellSpacing.y;
        g.z = (c1 - c0) / cellSpacing.z;

        // the field is held constant past the first and last cell centres
        if (uRaw.x != u.x)
        {
            g.x = 0.0;
        }

        if (uRaw.y != u.y)
        {
            g.y = 0.0;
        }

        if (uRaw.z != u.z)
        {
            g.z = 0.0;
        }

        return lerp(c0, c1, f.z);
    }
};

public struct ImplicitBody3D : IImplicitBody
{
    private ImplicitGrid3D nearGrid; // fine grid around the body (half body, y >= 0, if mirrorY is set)
    private ImplicitGrid3D farGrid;  // coarse grid around the near grid
    private float bodyRadius;
    private uint mirrorY;
    private uint useKelvin;
    private uint _padding;

    private bool insideGrid(ImplicitGrid3D grid, float3 q)
    {
        float3 lo = grid.getOrigin() + 0.5 * grid.getCellSpacing();
        float3 hi = grid.getOrigin() + grid.getExtent() - 0.5 * grid.getCellSpacing();
        if (mirrorY != 0)
        {
            lo.y = -1.0; // the grid starts on the symmetry plane; holding the first cell value is exact there
        }

        return all(q >= lo) && all(q <= hi);
    }

    public float value(float3 x)
    {
        float3 q = x;
        if (mirrorY != 0)
        {
            q.y = abs(q.y);
        }

        if (insideGrid(nearGrid, q))
        {
            // away from the wall a bound is enough: the filtered lookup, less its own error
            float3 h = nearGrid.getCellSpacing();
            float v = nearGrid.fast(q);
            if (abs(v) > 2.0 * max(h.x, max(h.y, h.z)))
            {
                return 0.98 * v;
            }

            float3 g;
            return nearGrid.precise(q, g);
        }

        float lowerBound = length(x) - bodyRadius;
        if (insideGrid(farGrid, q))
        {
            return max(0.98 * farGrid.fast(q), lowerBound);
        }

        return lowerBound;
    }

    public float3 gradient(float3 x)
    {
        float3 q = x;
        if (mirrorY != 0)
        {
            q.y = abs(q.y);
        }

        float3 h = insideGrid(nearGrid, q) ? nearGrid.getCellSpacing() : farGrid.getCellSpacing();
        float3 g;
        g.x = (value(x + float3(h.x, 0.0, 0.0)) - value(x - float3(h.x, 0.0, 0.0))) / (2.0 * h.x);
        g.y = (value(x + float3(0.0, h.y, 0.0)) - value(x - float3(0.0, h.y, 0.0))) / (2.0 * h.y);
        g.z = (value(x + float3(0.0, 0.0, h.z)) - value(x - float3(0.0, 0.0, h.z))) / (2.0 * h.z);

        return g;
    }

    public float valueAndGradient(float3 x, out float3 g)
    {
        float3 q = x;
        if (mirrorY != 0)
        {
            q.y = abs(q.y);
        }

        if (insideGrid(nearGrid, q))
        {
            float v = nearGrid.precise(q, g);
            if (mirrorY != 0 && x.y < 0.0)
            {
                g.y = -g.y;
            }

            return v;
        }

        g = gradient(x);
        return value(x);
    }

    public bool usesKelvin()
    {
        return useKelvin != 0;
    }

    public float boundingRadius()
    {
        return bodyRadius;
    }
};

public struct ImplicitReflectingBoundaryGeometricQueries<Body, let DIM : int> : IReflectingBoundaryGeometricQueries<DIM>
    where Body : IImplicitBody
{
    private Body body;

    /////////////////////////////////////////////////////////////////////////////////////////////
    // the implicit in walk space (y); the body lives in x = y / |y|^2 when the Kelvin transform is used

    private float3 toBodySpace(float3 y)
    {
        if (!body.usesKelvin())
        {
            return y;
        }

        return y / max(dot(y, y), 1e-20);
    }

    // maps a gradient of the field from body space to walk space
    private float3 toWalkSpaceGradient(float3 y, float3 g)
    {
        if (!body.usesKelvin())
        {
            return g;
        }

        float r2 = max(dot(y, y), 1e-20);
        float3 yHat = y / sqrt(r2);

        return (g - 2.0 * yHat * dot(yHat, g)) / r2;
    }

    // field value at a walk-space point: > 0 in the flow domain, zero on the wall
    private float rawValue(float3 y)
    {
        return body.value(toBodySpace(y));
    }

    // field value and the gradient of the interpolant in walk space
    private float rawValueAndGradient(float3 y, out float3 g)
    {
        float3 gBody;
        float v = body.valueAndGradient(toBodySpace(y), gBody);
        g = toWalkSpaceGradient(y, gBody);

        return v;
    }

    // conservative walk-space distance (same sign as the field value v at y)
    private float conservativeDistance(float3 y, float v)
    {
        if (!body.usesKelvin())
        {
            return v;
        }

        // the ball of radius |v| about x is free of wall; so is its image, a ball about y of this radius
        float r = length(y);
        float a = abs(v);
        float d = a * r * r / (1.0 + a * r);
        if (v > 0.0)
        {
            // the wall lies in |y| >= 1 / (radius of the body)
            d = max(d, 1.0 / body.boundingRadius() - r);
        }

        return v >= 0.0 ? d : -d;
    }

    // unit normal that points out of the flow domain (into the body); smooth gradient
    private float3 outwardNormal(float3 y)
    {
        float3 g = toWalkSpaceGradient(y, body.gradient(toBodySpace(y)));
        return -g / max(length(g), 1e-30);
    }

    // moves a point onto the wall (Newton steps along the gradient); returns the normal there
    private void projectToWall(inout float3 y, out float3 n)
    {
        for (int i = 0; i < 6; i++)
        {
            float3 g;
            float v = rawValueAndGradient(y, g);
            float g2 = dot(g, g);
            if (g2 < 1e-30)
            {
                break;
            }

            y -= g * (v / g2);
        }

        n = outwardNormal(y);
    }

    // finds the wall on the line q + t * n (Newton steps); lengthScale limits |t| and sets the tolerance;
    // nw is the unit normal of the interpolant at the point found
    private bool findWallOnLine(float3 q, float3 n, float lengthScale, out float3 w, out float3 nw, out float t)
    {
        t = 0.0;
        w = q;
        nw = n;
        float3 g;
        float v = rawValueAndGradient(q, g);
        for (int i = 0; i < 5; i++)
        {
            float slope = dot(g, n);
            if (abs(slope) < 1e-30)
            {
                return false;
            }

            t -= v / slope;
            if (abs(t) > lengthScale)
            {
                return false;
            }

            v = rawValueAndGradient(q + t * n, g);
        }

        w = q + t * n;
        float gLen = length(g);
        nw = -g / max(gLen, 1e-30);

        return abs(v) <= 0.01 * lengthScale * gLen;
    }

    private void tangentFrame(float3 n, out float3 e1, out float3 e2)
    {
        float s = n.z >= 0.0 ? 1.0 : -1.0;
        float a = -1.0 / (s + n.z);
        float b = n.x * n.y * a;
        e1 = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
        e2 = float3(b, s + n.y * n.y * a, -n.y);
    }

    // size of the Robin coefficient of the inverted problem at a wall point (kelvin-transform.slang, robinCoeff)
    private float robinCoeffSize(float3 w, float3 nw)
    {
        return body.usesKelvin() ? abs(dot(nw, w)) / max(dot(w, w), 1e-20) : 0.0;
    }

    // checks that the wall inside the ball of radius R about y is seen from one side only, as one sheet over the
    // tangent plane at the closest wall point p, and that the Robin reflectance stays within [0, 2] in the ball
    private bool starRegionIsValid(float3 y, float3 p, float3 n, float3 e1, float3 e2, float heightAboveWall, float R)
    {
        const int nAngles = 6;
        float hMin = heightAboveWall;
        float muMax = robinCoeffSize(p, n);
        for (int ring = 0; ring < 2; ring++)
        {
            float rho = ring == 0 ? 0.6 * R : R;
            for (int a = 0; a < nAngles; a++)
            {
                float phi = (float(a) + (ring == 0 ? 0.0 : 0.5)) * (2.0 * M_PI / float(nAngles));
                float3 q = p + rho * (cos(phi) * e1 + sin(phi) * e2);
                float3 w, nw;
                float t;
                if (!findWallOnLine(q, n, 0.5 * rho, w, nw, t))
                {
                    return false; // the wall ends, folds or stands too steep here
                }

                if (dot(nw, n) < 0.5)
                {
                    return false;
                }

                // height of y above the tangent plane at w: zero on a silhouette, negative behind it
                float h = dot(nw, w - y);
                if (h < 0.25 * heightAboveWall)
                {
                    return false;
                }

                hMin = min(hMin, h);
                muMax = max(muMax, robinCoeffSize(w, nw));
            }
        }

        // reflectance = 1 - mu r^2 (1 - r / R) / h for a wall point at distance r, and r^2 (1 - r / R) <= 4 R^2 / 27
        return muMax * (4.0 / 27.0) * R * R <= 0.9 * hMin;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    // IReflectingBoundaryGeometricQueries

    // computes the distance to the boundary (negative inside the flow domain when signed)
    public float computeDistToBoundary(vector<float, DIM> x,
                                       bool computeSignedDistance,
                                       float maxDistance = FLT_MAX)
    {
        float3 y = toFloat3<DIM>(x);
        float d = conservativeDistance(y, rawValue(y));

        return computeSignedDistance ? -d : abs(d);
    }

    // projects a point to the boundary
    public bool projectToBoundary(inout vector<float, DIM> x,
                                  out vector<float, DIM> normal,
                                  out float distance,
                                  bool computeSignedDistance)
    {
        float3 y = toFloat3<DIM>(x);
        float d = conservativeDistance(y, rawValue(y));
        float3 n;
        projectToWall(y, n);
        x = fromFloat3<DIM>(y);
        normal = fromFloat3<DIM>(n);
        distance = computeSignedDistance ? -d : abs(d);

        return true;
    }

    // intersects a ray with the boundary
    public bool intersectBoundary(vector<float, DIM> origin,
                                  vector<float, DIM> normal,
                                  vector<float, DIM> dir,
                                  float tMax,
                                  bool onBoundary,
                                  out IntersectionPoint<DIM> intersectionPt)
    {
        intersectionPt = IntersectionPoint<DIM>();
        float3 o = toFloat3<DIM>(origin);
        float3 d = toFloat3<DIM>(dir);

        // a ray that starts on the wall must start on the flow side of it
        float v = rawValue(o);
        if (onBoundary && v <= 0.0)
        {
            float3 n;
            projectToWall(o, n);
            o -= n * (2e-6 * max(1.0, length(o)));
            v = rawValue(o);
        }

        // sphere tracing; the smallest step is a fixed part of the ray, so the march always ends
        const int maxIterations = 420;
        float minStep = tMax / 400.0;
        float tPrev = 0.0;
        float t = 0.0;
        float vPrev = v;
        bool hit = false;
        for (int i = 0; i < maxIterations; i++)
        {
            if (v <= 0.0)
            {
                hit = true;
                break;
            }

            if (t >= tMax)
            {
                break;
            }

            float step = max(conservativeDistance(o + t * d, v), minStep);
            tPrev = t;
            vPrev = v;
            t = min(t + step, tMax);
            v = rawValue(o + t * d);
        }

        if (!hit)
        {
            return false;
        }

        // bisection on the sign of the field between the last point in the flow and the first point in the body
        float a = tPrev;
        float b = t;
        float va = vPrev;
        float vb = v;
        for (int j = 0; j < 12; j++)
        {
            float m = 0.5 * (a + b);
            float vm = rawValue(o + m * d);
            if (vm > 0.0)
            {
                a = m;
                va = vm;
            }
            else
            {
                b = m;
                vb = vm;
            }
        }

        // one linear step between the two ends; the hit point stays on the flow side
        if (va > 0.0 && vb < 0.0)
        {
            float m = a + (b - a) * va / (va - vb);
            if (rawValue(o + m * d) > 0.0)
            {
                a = m;
            }
        }

        float3 pHit = o + a * d;
        intersectionPt.pt = fromFloat3<DIM>(pHit);
        intersectionPt.normal = fromFloat3<DIM>(outwardNormal(pHit));
        intersectionPt.dist = length(pHit - toFloat3<DIM>(origin));
        return true;
    }

    // checks whether there is a line of sight between two points; wall samples come from sampleBoundary below,
    // which only returns points of the one visible sheet inside a valid star region
    public bool intersectsWithBoundary(vector<float, DIM> xi,
                                       vector<float, DIM> xj,
                                       vector<float, DIM> ni,
                                       vector<float, DIM> nj,
                                       bool offseti,
                                       bool offsetj)
    {
        return false;
    }

    // samples a point on the boundary inside the ball of the given radius about x
    public bool sampleBoundary(vector<float, DIM> x,
                               float radius,
                               vector<float, DIM> randNums,
                               out BoundarySample<DIM> boundarySample)
    {
        boundarySample = BoundarySample<DIM>();
        float3 y = toFloat3<DIM>(x);
        float d = conservativeDistance(y, rawValue(y));
        if (d >= radius)
        {
            return false; // no wall inside the ball
        }

        // closest wall point and tangent plane
        float3 p = y;
        float3 n;
        projectToWall(p, n);
        float height = max(dot(p - y, n), 0.0);
        if (height >= radius)
        {
            return false;
        }

        // point on the tangent disc, with a density close to the Green's function of the ball (1 / distance)
        float rMax = sqrt(radius * radius + height * height);
        float r = height + randNums[0] * (rMax - height);
        float rho = sqrt(max(r * r - height * height, 0.0));
        float phi = 2.0 * M_PI * randNums[1];
        float3 e1, e2;
        tangentFrame(n, e1, e2);
        float3 q = p + rho * (cos(phi) * e1 + sin(phi) * e2);

        // move it onto the wall along the normal
        float3 w, nRough;
        float t;
        if (!findWallOnLine(q, n, max(0.5 * rho, 0.05 * radius), w, nRough, t))
        {
            return false;
        }

        float3 nw = outwardNormal(w);
        float cosTilt = dot(nw, n);
        if (cosTilt < 0.2)
        {
            return false;
        }

        float pdfPlane = 1.0 / (2.0 * M_PI * max(r, 1e-30) * (rMax - height));
        boundarySample.pt = fromFloat3<DIM>(w);
        boundarySample.normal = fromFloat3<DIM>(nw);
        boundarySample.pdf = pdfPlane * cosTilt;
        return true;
    }

    // computes the radius of a star-shaped region on the boundary
    public float computeStarRadiusForBoundary(vector<float, DIM> x,
                                              float minRadius,
                                              float maxRadius,
                                              float silhouettePrecision,
                                              bool flipNormalOrientation)
    {
        if (minRadius > maxRadius)
        {
            return maxRadius;
        }

        float3 y = toFloat3<DIM>(x);
        float d = conservativeDistance(y, rawValue(y));
        if (d >= maxRadius)
        {
            return maxRadius; // the ball cannot reach the wall
        }

        if (d <= 0.0)
        {
            return minRadius;
        }

        // closest wall point; the ball of radius d about y is always valid
        float3 p = y;
        float3 n;
        projectToWall(p, n);
        float height = dot(p - y, n);
        float best = d;
        if (height > 0.0 && height < 4.0 * d)
        {
            float3 e1, e2;
            tangentFrame(n, e1, e2);
            float R = 2.0 * d;
            for (int k = 0; k < 8; k++)
            {
                float Rk = min(R, maxRadius);
                if (!starRegionIsValid(y, p, n, e1, e2, height, Rk))
                {
                    break;
                }

                best = Rk;
                if (Rk >= maxRadius)
                {
                    break;
                }

                R *= 2.0;
            }
        }

        return max(best, minRadius);
    }
};

// boundary samples for boundary value caching from a list of points (position, normal, pdf per unit area),
// for walls that have no triangles; the samples are stratified along the list
public struct PointListBoundarySampler<let DIM : int> : IBoundarySampler<DIM>
{
    private StructuredBuffer<float> samples; // 7 floats per sample: position, normal, pdf
    private uint nListSamples;
    private float3 domainMin;
    private float3 domainMax;

    // generates a sample point on the boundary
    public void generateSample(SampleType sampleType,
                               float normalOffsetForBoundary,
                               uint generateBoundaryNormalAligned,
                               uint sampleIndex,
                               uint nTotalSamples,
                               inout pcg32 rng,
                               inout SamplePoint<DIM> samplePt)
    {
        // one stratum of the list per sample: the list is in a space-filling order, each entry stands for the
        // same wall area, so the samples spread evenly over the wall
        float u = (float(sampleIndex) + rng.nextFloat()) / float(max(nTotalSamples, 1));
        uint j = min(uint(u * float(nListSamples)), nListSamples - 1);
        float3 pt = float3(samples[7 * j], samples[7 * j + 1], samples[7 * j + 2]);
        float3 n = float3(samples[7 * j + 3], samples[7 * j + 4], samples[7 * j + 5]);
        samplePt.pt = fromFloat3<DIM>(pt);
        samplePt.normal = fromFloat3<DIM>(n);
        samplePt.type = sampleType;
        samplePt.estimationQuantity = EstimationQuantity::Solution;
        samplePt.pdf = samples[7 * j + 6];
        samplePt.distToAbsorbingBoundary = length(min(domainMin - pt, pt - domainMax));
        samplePt.distToReflectingBoundary = 0.0;
        samplePt.estimateBoundaryNormalAligned = 0;
    }
};

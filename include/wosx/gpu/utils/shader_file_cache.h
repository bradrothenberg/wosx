/*
 * ntop_aero addition to WoSX (this fork): a shader cache on disk.
 *
 * WoSX compiles its Slang shaders every time a solver is made. With the environment variable WOSX_SHADER_CACHE set
 * to a folder, the compiled code of every entry point is kept there, keyed by the hash that Slang gives it (source,
 * macros, target), and read back on the next run. With the variable unset nothing changes.
 */

#pragma once

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <slang-rhi.h>

namespace wosx {

class ShaderCacheBlob : public ISlangBlob {
public:
    explicit ShaderCacheBlob(std::vector<char>&& data_): data(std::move(data_)) {}
    virtual ~ShaderCacheBlob() = default;

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) override {
        if (uuid == ISlangUnknown::getTypeGuid() || uuid == ISlangBlob::getTypeGuid()) {
            addRef();
            *outObject = static_cast<ISlangBlob*>(this);
            return SLANG_OK;
        }

        *outObject = nullptr;
        return SLANG_E_NO_INTERFACE;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return ++refCount; }

    SLANG_NO_THROW uint32_t SLANG_MCALL release() override {
        uint32_t count = --refCount;
        if (count == 0) delete this;
        return count;
    }

    SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() override { return data.data(); }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override { return data.size(); }

private:
    std::atomic<uint32_t> refCount{0};
    std::vector<char> data;
};

class ShaderFileCache : public rhi::IPersistentCache {
public:
    // returns the cache named by WOSX_SHADER_CACHE, or nullptr if the variable is not set; it lives until the
    // process ends, so devices can hold a plain pointer to it
    static ShaderFileCache* fromEnvironment() {
        const char* dir = std::getenv("WOSX_SHADER_CACHE");
        if (!dir || !dir[0]) return nullptr;
        static ShaderFileCache* cache = new ShaderFileCache(dir);
        return cache;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const& uuid, void** outObject) override {
        if (uuid == ISlangUnknown::getTypeGuid() || uuid == rhi::IPersistentCache::getTypeGuid()) {
            *outObject = static_cast<rhi::IPersistentCache*>(this);
            return SLANG_OK;
        }

        *outObject = nullptr;
        return SLANG_E_NO_INTERFACE;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 1; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 1; }

    SLANG_NO_THROW rhi::Result SLANG_MCALL writeCache(ISlangBlob* key, ISlangBlob* data) override {
        if (!key || !data) return SLANG_FAIL;
        std::filesystem::path file = fileOf(key);
        std::filesystem::path tmp = file;
        tmp += "." + std::to_string(++writeCount) + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary);
            if (!out) return SLANG_FAIL;
            out.write((const char*)data->getBufferPointer(), (std::streamsize)data->getBufferSize());
            if (!out) return SLANG_FAIL;
        }

        std::error_code ec;
        std::filesystem::rename(tmp, file, ec);
        if (ec) std::filesystem::remove(tmp, ec);
        return SLANG_OK;
    }

    SLANG_NO_THROW rhi::Result SLANG_MCALL queryCache(ISlangBlob* key, ISlangBlob** outData) override {
        if (!key || !outData) return SLANG_FAIL;
        std::ifstream in(fileOf(key), std::ios::binary | std::ios::ate);
        if (!in) return SLANG_E_NOT_FOUND;
        std::streamsize size = in.tellg();
        if (size <= 0) return SLANG_E_NOT_FOUND;
        std::vector<char> data((size_t)size);
        in.seekg(0);
        in.read(data.data(), size);
        if (!in) return SLANG_E_NOT_FOUND;
        ShaderCacheBlob* blob = new ShaderCacheBlob(std::move(data));
        blob->addRef();
        *outData = blob;
        ++hitCount;
        return SLANG_OK;
    }

    uint32_t hits() const { return hitCount; }

private:
    explicit ShaderFileCache(const std::string& dir_): dir(dir_) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
    }

    std::filesystem::path fileOf(ISlangBlob* key) const {
        static const char* digits = "0123456789abcdef";
        const unsigned char* bytes = (const unsigned char*)key->getBufferPointer();
        size_t n = key->getBufferSize();
        std::string name;
        name.reserve(2*n + 4);
        for (size_t i = 0; i < n; i++) {
            name.push_back(digits[bytes[i] >> 4]);
            name.push_back(digits[bytes[i] & 15]);
        }

        return dir / (name + ".bin");
    }

    std::filesystem::path dir;
    std::atomic<uint32_t> writeCount{0};
    std::atomic<uint32_t> hitCount{0};
};

} // namespace wosx

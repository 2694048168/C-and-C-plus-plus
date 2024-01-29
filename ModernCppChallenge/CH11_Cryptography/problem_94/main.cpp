/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief File signing
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "aes.h"
#include "files.h"
#include "hex.h"
#include "osrng.h"
#include "rsa.h"

#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

/**
 * @brief File signing
 * 
 * Write a program that is able to sign files and verify that a signed file 
 * has not been tampered with, using RSA cryptography. When signing a file, 
 * the signature should be written to a separate file and used later for the verification
 *  process. The program should provide at least two functions: one that signs a file
 * (taking as arguments the path to the file, the path to the RSA private key, 
 * and the path to the file where the signature will be written) and one that
 * verifies a file (taking as arguments the path to the file, the path to the RSA
 *  public key, and the path to the signature file).
 * 
 */

/**
 * @brief Solution: 
------------------------------------------------------ */
void encode(const fs::path &filepath, const CryptoPP::BufferedTransformation &bt)
{
    CryptoPP::FileSink file(filepath.c_str());

    bt.CopyTo(file);
    file.MessageEnd();
}

void encode_private_key(const fs::path &filepath, const CryptoPP::RSA::PrivateKey &key)
{
    CryptoPP::ByteQueue queue;
    key.DEREncodePrivateKey(queue);

    encode(filepath, queue);
}

void encode_public_key(const fs::path &filepath, const CryptoPP::RSA::PublicKey &key)
{
    CryptoPP::ByteQueue queue;
    key.DEREncodePublicKey(queue);

    encode(filepath, queue);
}

void decode(const fs::path &filepath, CryptoPP::BufferedTransformation &bt)
{
    CryptoPP::FileSource file(filepath.c_str(), true);

    file.TransferTo(bt);
    bt.MessageEnd();
}

void decode_private_key(const fs::path &filepath, CryptoPP::RSA::PrivateKey &key)
{
    CryptoPP::ByteQueue queue;

    decode(filepath, queue);
    key.BERDecodePrivateKey(queue, false, queue.MaxRetrievable());
}

void decode_public_key(const fs::path &filepath, CryptoPP::RSA::PublicKey &key)
{
    CryptoPP::ByteQueue queue;

    decode(filepath, queue);
    key.BERDecodePublicKey(queue, false, queue.MaxRetrievable());
}

void rsa_sign_file(const fs::path &filepath, const fs::path &privateKeyPath, const fs::path &signaturePath,
                   CryptoPP::RandomNumberGenerator &rng)
{
    CryptoPP::RSA::PrivateKey privateKey;
    decode_private_key(privateKeyPath, privateKey);

    CryptoPP::RSASSA_PKCS1v15_SHA_Signer signer(privateKey);

    CryptoPP::FileSource fileSource(
        filepath.c_str(), true, new CryptoPP::SignerFilter(rng, signer, new CryptoPP::FileSink(signaturePath.c_str())));
}

bool rsa_verify_file(const fs::path &filepath, const fs::path &publicKeyPath, const fs::path &signaturePath)
{
    CryptoPP::RSA::PublicKey publicKey;
    decode_public_key(publicKeyPath.c_str(), publicKey);

    CryptoPP::RSASSA_PKCS1v15_SHA_Verifier verifier(publicKey);

    CryptoPP::FileSource signatureFile(signaturePath.c_str(), true);

    if (signatureFile.MaxRetrievable() != verifier.SignatureLength())
        return false;

    CryptoPP::SecByteBlock signature(verifier.SignatureLength());
    signatureFile.Get(signature, signature.size());

    auto *verifierFilter = new CryptoPP::SignatureVerificationFilter(verifier);
    verifierFilter->Put(signature, verifier.SignatureLength());

    CryptoPP::FileSource fileSource(filepath.c_str(), true, verifierFilter);

    return verifierFilter->GetLastResult();
}

void generate_keys(const fs::path &privateKeyPath, const fs::path &publicKeyPath, CryptoPP::RandomNumberGenerator &rng)
{
    try
    {
        CryptoPP::RSA::PrivateKey rsaPrivate;
        rsaPrivate.GenerateRandomWithKeySize(rng, 3072);

        CryptoPP::RSA::PublicKey rsaPublic(rsaPrivate);

        encode_private_key(privateKeyPath, rsaPrivate);
        encode_public_key(publicKeyPath, rsaPublic);
    }
    catch (const CryptoPP::Exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    CryptoPP::AutoSeededRandomPool rng;

    generate_keys("rsa-private.key", "rsa-public.key", rng);

    rsa_sign_file("sample.txt", "rsa-private.key", "sample.sign", rng);

    auto success = rsa_verify_file("sample.txt", "rsa-public.key", "sample.sign");

    assert(success);

    return 0;
}

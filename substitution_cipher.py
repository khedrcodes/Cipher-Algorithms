class SubstitutionCipher:
    def __init__(self, key):
        self.key = key.lower()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext.lower():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_char = self.key[index]
                ciphertext += new_char
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext.lower():
            if char in self.key:
                index = self.key.index(char)
                new_char = self.alphabet[index]
                plaintext += new_char
            else:
                plaintext += char
        return plaintext


cipher = SubstitutionCipher("xnyahpogzqwbtsflrcvmuekjdi")
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

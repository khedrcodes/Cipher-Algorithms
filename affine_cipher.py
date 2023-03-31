class AffineCipher:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext.lower():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_index = (self.a * index + self.b) % 26
                ciphertext += self.alphabet[new_index]
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        a_inv = self.mod_inverse(self.a, 26)
        for char in ciphertext.lower():
            if char in self.alphabet:
                index = self.alphabet.index(char)
                new_index = (a_inv * (index - self.b)) % 26
                plaintext += self.alphabet[new_index]
            else:
                plaintext += char
        return plaintext

    def mod_inverse(self, a, m):
        for i in range(1, m):
            if (a * i) % m == 1:
                return i
        return -1


cipher = AffineCipher(7, 3)
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

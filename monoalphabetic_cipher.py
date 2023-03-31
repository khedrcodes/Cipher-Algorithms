class MonoalphabeticCipher:
    def __init__(self, key):
        new_key = ""
        alphapet = "abcdefghijklmnopqrstuvwxyz"
        for char in key:
            if not char in new_key:
                new_key += char
        for char in alphapet:
            if not char in new_key:
                new_key += char
        self.key = new_key

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext:
            if char.isalpha():
                ciphertext += self.key[ord(char) - 97]
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext.lower():
            if char.isalpha():
                plaintext += chr(self.key.index(char) + 97)
            else:
                plaintext += char
        return plaintext


cipher = MonoalphabeticCipher("example")
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

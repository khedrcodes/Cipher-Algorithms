class ShiftCipher:
    def __init__(self, shift):
        self.shift = shift

    def encrypt(self, plaintext):
        ciphertext = ""
        for char in plaintext:
            if char.isupper():
                ciphertext += chr((ord(char) + self.shift - 65) % 26 + 65)
            elif char.islower():
                ciphertext += chr((ord(char) + self.shift - 97) % 26 + 97)
            else:
                ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        for char in ciphertext:
            if char.isupper():
                plaintext += chr((ord(char) - self.shift - 65) % 26 + 65)
            elif char.islower():
                plaintext += chr((ord(char) - self.shift - 97) % 26 + 97)
            else:
                plaintext += char
        return plaintext


cipher = ShiftCipher(11)
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

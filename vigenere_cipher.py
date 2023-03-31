class VigenereCipher:
    def __init__(self, keyword):
        self.keyword = keyword

    def encrypt(self, plaintext):
        ciphertext = ""
        keyword_index = 0
        for char in plaintext:
            if char.isalpha():
                char = char.upper()
                keyword_char = self.keyword[keyword_index].upper()
                keyword_index = (keyword_index + 1) % len(self.keyword)
                shift = ord(keyword_char) - ord("A")
                char = chr(((ord(char) - ord("A") + shift) % 26) + ord("A"))
            ciphertext += char
        return ciphertext

    def decrypt(self, ciphertext):
        plaintext = ""
        keyword_index = 0
        for char in ciphertext:
            if char.isalpha():
                char = char.upper()
                keyword_char = self.keyword[keyword_index].upper()
                keyword_index = (keyword_index + 1) % len(self.keyword)
                shift = ord(keyword_char) - ord("A")
                char = chr(((ord(char) - ord("A") - shift + 26) % 26) + ord("A"))
            plaintext += char
        return plaintext


cipher = VigenereCipher("example")
plaintext = "mohamed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

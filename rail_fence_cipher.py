class RailFenceCipher:
    def encrypt(self, plain_text, rails):
        fence = [["\n" for i in range(len(plain_text))] for j in range(rails)]
        direction = -1
        row, col = 0, 0
        for char in plain_text:
            if row == 0 or row == rails - 1:
                direction *= -1
            fence[row][col] = char
            col += 1
            row += direction
        result = []
        for i in range(rails):
            for j in range(len(plain_text)):
                if fence[i][j] != "\n":
                    result.append(fence[i][j])
        return "".join(result)

    def decrypt(self, cipher_text, rails):
        fence = [["\n" for i in range(len(cipher_text))] for j in range(rails)]
        direction = -1
        row, col = 0, 0
        for i in range(len(cipher_text)):
            if row == 0 or row == rails - 1:
                direction *= -1
            fence[row][col] = "*"
            col += 1
            row += direction
        index = 0
        for i in range(rails):
            for j in range(len(cipher_text)):
                if (fence[i][j] == "*") and (index < len(cipher_text)):
                    fence[i][j] = cipher_text[index]
                    index += 1
        result = []
        row, col = 0, 0
        for i in range(len(cipher_text)):
            if row == 0 or row == rails - 1:
                direction *= -1
            if fence[row][col] != "*":
                result.append(fence[row][col])
                col += 1
            row += direction
        return "".join(result)


cipher = RailFenceCipher()
plaintext = "mohamedmahmoudkhedr"
rails = 3
ciphertext = cipher.encrypt(plaintext, rails)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext, rails)
print("plaintext = ", decrypted_plaintext)

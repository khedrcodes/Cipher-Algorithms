class PlayfairCipher:
    def __init__(self, key):
        self.key = key
        self.key_matrix = self._generate_key_matrix()

    def encrypt(self, plaintext):
        pairs = self._prepare_input(plaintext)
        ciphertext = ""
        for pair in pairs:
            ciphertext += self._encrypt_pair(pair)
        return ciphertext

    def decrypt(self, ciphertext):
        pairs = self._prepare_input(ciphertext)
        plaintext = ""
        for pair in pairs:
            plaintext += self._decrypt_pair(pair)
        return plaintext

    def _generate_key_matrix(self):
        key_matrix = []
        key_chars = []
        for char in self.key:
            if char not in key_chars:
                key_chars.append(char)
        for i in range(ord("a"), ord("z") + 1):
            char = chr(i)
            if char == "j":
                continue
            if char not in key_chars:
                key_chars.append(char)
        for i in range(0, 25, 5):
            row = key_chars[i : i + 5]
            key_matrix.append(row)
        return key_matrix

    def _prepare_input(self, text):
        text = text.lower().replace(" ", "").replace("j", "i")
        if len(text) % 2 == 1:
            text += "x"
        pairs = []
        i = 0
        while i < len(text):
            pair = text[i : i + 2]
            pairs.append(pair)
            i += 2
        return pairs

    def _encrypt_pair(self, pair):
        char1, char2 = pair[0], pair[1]
        row1, col1 = self._find_char(char1)
        row2, col2 = self._find_char(char2)
        if row1 == row2:
            return (
                self.key_matrix[row1][(col1 + 1) % 5]
                + self.key_matrix[row2][(col2 + 1) % 5]
            )
        elif col1 == col2:
            return (
                self.key_matrix[(row1 + 1) % 5][col1]
                + self.key_matrix[(row2 + 1) % 5][col2]
            )
        else:
            return self.key_matrix[row1][col2] + self.key_matrix[row2][col1]

    def _decrypt_pair(self, pair):
        char1, char2 = pair[0], pair[1]
        row1, col1 = self._find_char(char1)
        row2, col2 = self._find_char(char2)
        if row1 == row2:
            return (
                self.key_matrix[row1][(col1 - 1) % 5]
                + self.key_matrix[row2][(col2 - 1) % 5]
            )
        elif col1 == col2:
            return (
                self.key_matrix[(row1 - 1) % 5][col1]
                + self.key_matrix[(row2 - 1) % 5][col2]
            )
        else:
            return self.key_matrix[row1][col2] + self.key_matrix[row2][col1]

    def _find_char(self, char):
        for i in range(5):
            for j in range(5):
                if self.key_matrix[i][j] == char:
                    return i, j


cipher = PlayfairCipher("monarchy")
plaintext = "mohammed mahmoud khedr"
ciphertext = cipher.encrypt(plaintext)
print("ciphertext = ", ciphertext)
decrypted_plaintext = cipher.decrypt(ciphertext)
print("plaintext = ", decrypted_plaintext)

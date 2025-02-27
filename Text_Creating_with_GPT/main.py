from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from arayuz import Ui_Form
import sys
import torch

model_name = "gpt2"

# Tokenizer ve model oluşturuluyor
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        
        # Buton tıklama sinyali ile fonksiyon bağlantısı
        self.ui.pushButton.clicked.connect(self.generate_text)

    def generate_text(self):
        # lineEdit'ten metni al
        input_text = self.ui.lineEdit.text()

        # GPT-2 ile metin üretimi
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

        # Üretilen metni çözümle ve lineEdit_2'ye yaz
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.ui.lineEdit_2.setText(generated_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

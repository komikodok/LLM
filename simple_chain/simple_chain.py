from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

import json
import os


with open("config.json", "r") as f:
    config = json.load(f)
    
model_config = config["model"]

llm = ChatOllama(model=model_config["repo_id"])
template = '''
            You're name is Ruby. \
            You're expert AI assistant for given a context about user question. \
            Don't answer anything except context about user question. \
        '''
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{question}")
    ]
)

chain = prompt | llm
result = chain.invoke({'question':'Perkembangan teknologi otomatisasi pada penghapus papan tulis dapat meningkatkan tingkat kebersihandan kesehatan, karena dengan sistem otomatisasikegiatan menghapus papan tulis dapat dijalankan secaraotomatis tanpa menghirup tinta spidol yang berdampak pada kesehatan dan tidak mengotori tangan. Tujuan dan mekanisme kerja pembuatan alat penghapus papan tulis otomatis dengan menggunakan remote dan arduino. Metode perancangan penghapus papan tulis otomatis dibuat dengan menggunakan arduino uno sebagai mikrokontroler dan infrared remote untuk pengendali motor stepper menggerakkan lengan penghapus kekanan dan kekiri. Pembuatan alat ini dilakukan dengan menyiapkan bahan seperti papan tulis yang sudah dibuatkan rangka. Kemudian, pemasangan alumunim V slot sebagai rel atas dan rel bawah. kemudian pemasangan roda atas dan bawah pada alumunium V slot rel atas bawah yang telah dipasangkan pada lengan penghapus. Hasil perancangan alat penghapus papan tulis otomatis ini, mampu memudahkan pengajar maupun pelajar dalam proses penghapusan papan tulis. Kesimpulan pembuatan alat ini yaitu penghapus papan tulis otomatis dapat bekerja apabila ketika inrared remote ditekan maka penghapus papan tulis akan bergerak kekanan dan kekiri sesuai dengan program yang telah dimasukkan kedalam arduinouno. mekanisme kerja penghapus papan tulis ini berfungsi dengan baik untuk menghapus tinta spidol pada papan tulis dengan daya sebesar 25,60 watt.'})
print(result)



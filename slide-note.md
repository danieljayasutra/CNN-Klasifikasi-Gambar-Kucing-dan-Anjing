## PENDAHULUAN :

1. Pengenalan tentang masalah klasifikasi gambar kucing dan anjing.
Klasifikasi gambar kucing dan anjing adalah salah satu masalah klasifikasi biner yang umum dalam pengolahan gambar.
Tujuannya adalah untuk mengembangkan model yang dapat membedakan antara gambar kucing dan gambar anjing secara otomatis

2. Pentingnya Penggunaan Convolutional Neural Network (CNN) dalam Pengolahan Gambar.
Convolutional Neural Network (CNN) adalah arsitektur neural network yang dirancang khusus untuk pengolahan gambar.
CNN memiliki kemampuan untuk mengekstrak fitur-fitur penting dari gambar secara hierarkis, yang memungkinkan mereka untuk memahami struktur spasial dan pola dalam gambar.
Keberhasilan CNN dalam tugas-tugas pengolahan gambar telah membuatnya menjadi pilihan utama dalam berbagai aplikasi, seperti klasifikasi gambar, deteksi objek, dan segmentasi gambar.

3. Tujuan dari Proyek Ini
Tujuan utama dari proyek ini adalah untuk mengembangkan model CNN yang dapat membedakan antara gambar kucing dan anjing dengan tingkat akurasi yang tinggi.
Selain itu, proyek ini bertujuan untuk memperkenalkan konsep-konsep dasar dalam pembangunan dan pelatihan model CNN menggunakan bahasa pemrograman Python dan framework Keras.
Dengan menyelesaikan proyek ini, diharapkan kita dapat memperoleh pemahaman yang lebih baik tentang bagaimana CNN bekerja dalam konteks pengolahan gambar dan bagaimana menerapkannya dalam memecahkan masalah klasifikasi gambar yang konkret.



## PENDAHULUAN KERAS DAN OPEN CV :

1. Pengantar Singkat tentang Keras dan OpenCV
Keras: Keras adalah sebuah library open-source yang menyediakan antarmuka yang mudah digunakan dan efisien untuk membangun, melatih, dan mengevaluasi model neural network di Python. Keras dikembangkan dengan fokus pada kesederhanaan, modularitas, dan ekstensibilitas, sehingga cocok untuk pemula maupun peneliti yang ingin melakukan eksplorasi dalam pembangunan model neural network.
OpenCV: OpenCV (Open Source Computer Vision Library) adalah library open-source yang berfokus pada pemrosesan gambar dan vision komputer. 
OpenCV menyediakan berbagai fungsi dan algoritma untuk memanipulasi, mengolah, dan menganalisis gambar dan video. Dengan fitur-fitur seperti deteksi objek, segmentasi gambar, dan pelacakan objek, OpenCV menjadi salah satu pilihan utama dalam pengembangan aplikasi vision komputer.


2. Pentingnya Keras dalam Pembangunan Model Neural Network:
Keras menawarkan antarmuka yang sederhana dan intuitif untuk membangun dan melatih model neural network. Hal ini membuatnya menjadi pilihan yang populer bagi para pengembang yang ingin cepat memulai dalam pembangunan model.
Keras menyediakan dukungan untuk berbagai jenis arsitektur neural network, termasuk CNN, LSTM, dan model-model yang lebih kompleks.
Keras memiliki backend yang dapat disesuaikan, yang berarti Anda dapat menggunakan library back-end seperti TensorFlow, Theano, atau Microsoft Cognitive Toolkit (CNTK) sebagai "engine" untuk menjalankan model-model yang Anda buat menggunakan Keras.


3. Penggunaan OpenCV untuk Memanipulasi dan Memproses Gambar:
Dalam konteks proyek klasifikasi gambar kucing dan anjing, OpenCV digunakan untuk membaca gambar dari direktori, mengubah warna, dan memproses gambar sebelum diberikan ke model neural network untuk pelatihan atau inferensi.
OpenCV menyediakan berbagai fungsi dan algoritma untuk memanipulasi dan memproses gambar secara efisien. Ini termasuk operasi dasar seperti pembacaan, penulisan, dan konversi format gambar, serta operasi lanjutan seperti deteksi tepi, segmentasi, dan pencocokan fitur.
Kemampuan OpenCV untuk memanipulasi gambar dengan cepat dan efisien menjadikannya salah satu pilihan utama dalam pengolahan gambar dan vision komputer.



## ARSITEKTUR MODEL CNN:

1. Penjelasan Mengenai Arsitektur Model CNN yang Digunakan:
Arsitektur model CNN yang digunakan dalam proyek ini adalah sebuah model sederhana yang terdiri dari beberapa lapisan konvolusi (Conv2D) yang diikuti oleh lapisan pengecilan (MaxPooling2D).
Tujuan dari arsitektur ini adalah untuk secara bertahap mengekstrak fitur-fitur penting dari gambar secara hierarkis, kemudian mengurangi dimensi data untuk mengurangi kompleksitas dan mempercepat proses pelatihan.
Arsitektur ini biasanya digunakan sebagai baseline atau awal dalam pembangunan model CNN untuk tugas-tugas klasifikasi gambar yang sederhana.

2. Detail Setiap Lapisan dalam Model CNN:
Lapisan Conv2D: Lapisan konvolusi melakukan proses konvolusi pada input gambar menggunakan filter (kernel) untuk mengekstrak fitur-fitur gambar yang penting.
Lapisan MaxPooling2D: Lapisan pengecilan (pooling) digunakan untuk mengurangi dimensi spasial dari representasi gambar dan jumlah parameter dalam model. Ini dilakukan dengan memilih nilai maksimum dari setiap jendela berukuran kecil pada representasi gambar.
Lapisan Flatten: Lapisan ini digunakan untuk mengubah matriks 2D hasil dari lapisan-lapisan sebelumnya menjadi vektor 1D yang dapat diolah oleh lapisan-lapisan Dense.
Lapisan Dense: Lapisan-lapisan fully connected (sepenuhnya terhubung) digunakan untuk melakukan klasifikasi berdasarkan fitur-fitur yang telah diekstrak sebelumnya.

3. Alasan Penggunaan Fungsi Aktivasi dan Dropout:
Fungsi Aktivasi: Fungsi aktivasi, seperti ReLU (Rectified Linear Activation) digunakan untuk menambahkan non-linearitas ke dalam model. Ini memungkinkan model untuk belajar relasi yang lebih kompleks antara fitur-fitur input dan output. ReLU dipilih karena sederhana, efisien dalam komputasi, dan efektif dalam mencegah masalah vanish gradient.
Dropout: Dropout adalah teknik regularisasi yang digunakan untuk mencegah overfitting. Dengan dropout, sebagian unit (neuron) dalam lapisan yang diberikan secara acak dimatikan (dropout) selama proses pelatihan. Hal ini memaksa model untuk belajar fitur-fitur yang lebih robust dan mencegahnya untuk menjadi terlalu bergantung pada fitur-fitur spesifik dari data pelatihan. Dropout juga membantu mencegah ketergantungan antara unit-unit dalam model, sehingga meningkatkan generalisasi.

## PELATIHAN MODEL

1. Proses pelatihan model menggunakan generator gambar
Proses pelatihan model menggunakan generator gambar memungkinkan kita untuk memuat dan memproses gambar secara batch secara bertahap, yang berguna saat kita memiliki dataset gambar yang besar dan tidak muat dalam memori.
Dalam keras, kita menggunakan fit_generator() untuk melatih model menggunakan generator gambar. Generator gambar dapat dibuat menggunakan ImageDataGenerator() yang disediakan oleh Keras.
Contoh Kode:
from keras.preprocessing.image import ImageDataGenerator
```
# Membuat generator gambar untuk data pelatihan
train_image_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Menggunakan generator gambar untuk memuat dan memproses data pelatihan
train_generator = train_image_gen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Melatih model menggunakan generator gambar
model.fit_generator(
    train_generator,
    steps_per_epoch=1000 // 32,  # jumlah batch yang akan dieksekusi pada setiap epoch
    epochs=10
)
```

2. Penggunaan batch size dan jumlah epoch yang sesuai
Batch size adalah jumlah sampel yang diberikan kepada model dalam satu iterasi.
Jumlah epoch adalah jumlah kali keseluruhan dataset digunakan untuk melatih model.
Pemilihan batch size dan jumlah epoch yang sesuai dapat mempengaruhi kecepatan dan kualitas pelatihan model.
Contoh Kode:
```
# Pemilihan batch size dan jumlah epoch
batch_size = 32
epochs = 10

# Melatih model dengan menggunakan batch size dan jumlah epoch yang sesuai
model.fit(
    train_images,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(validation_images, validation_labels)
)
```

3. Evaluasi metrik seperti loss function dan akurasi:
Loss function digunakan untuk mengukur seberapa baik model memprediksi nilai target.
Akurasi adalah metrik yang mengukur seberapa baik model dalam memprediksi kelas target secara benar.
Dalam contoh kode, validation_images dan validation_labels adalah data validasi yang digunakan untuk evaluasi model.
Contoh Kode:
```
# Evaluasi model menggunakan data validasi
loss, accuracy = model.evaluate(validation_images, validation_labels)

# Menampilkan hasil evaluasi
print("Loss:", loss)
print("Accuracy:", accuracy)
```

## EVALUASI MODEL
1. Pengujian model pada data pengujian yang tidak terlihat sebelumnya:
Pengujian model pada data pengujian yang tidak terlihat sebelumnya penting untuk mengevaluasi kinerja model di luar data yang digunakan untuk pelatihan dan validasi.
Data pengujian seharusnya terpisah sepenuhnya dari data pelatihan dan validasi untuk memastikan bahwa model tidak "menghafal" atau "memori" data pelatihan.
Contoh Kode:
```
# Melakukan prediksi pada data pengujian
predictions = model.predict(test_images)

# Menampilkan hasil prediksi
print(predictions)
```

2. Interpretasi hasil prediksi model:
* model.predict(dog_img): Kode ini memanggil metode predict dari model yang telah dilatih sebelumnya dengan gambar dog_img sebagai argumennya. Metode ini digunakan untuk memprediksi kelas dari gambar yang diberikan.
* array([[0.6940112]], dtype=float32): Ini adalah output dari metode predict. 
Output ini berupa array numpy yang berisi nilai probabilitas yang diberikan oleh model untuk kelas positif (dalam kasus ini, kelas "DOG"). Nilai probabilitas ini menunjukkan seberapa yakin model dalam memprediksi bahwa gambar tersebut termasuk dalam kelas yang diinginkan ("DOG" dalam hal ini).
```# Untuk memeriksa akurasi
model.predict(dog_img)
array([[0.6940112]], dtype=float32)
```
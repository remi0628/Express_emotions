# Emotional expressive machine
時にはビデオ会議で顔を出したくない...そんな日はありませんか？

これを使用すればビデオ会議で顔を出さずに感情を表現することや表情に合ったリアクションを表示してくれます！！

# Features
* Pythonとリアルタイムビデオを使用してあなたの顔の感情を読み取り、リアクションを表示します
* ソフトウェアライブラリ「Dlib」を使用し、顔の傾きによって画像の傾きも変化します
* OBS studioと合わせて使用することでビデオ会議でスムーズに機能の切り替えが可能となります

**主な機能は以下の3つです**

1.Key[Z]:感情に合わせた画像の表示

2.Key[X]:カメラ使用時、自分の顔に合わせた画像を貼り付け

3.Key[C]:感情に合わせたリアクションを顔の周りに表示

※Key[Q]:終了
  - Keyは反応するまで長押ししてください

# Requirement
* Python3.5
* keras 2.0.5
* tensorflow 1.1.0
* opencv-python 3.2.0.6
* pandas 0.19.1
* PIL 7.1.2
* numpy 1.13.3
* scipy 1.1.0
* h5py 2.7.0
* statistics 1.0.3.5
* Dlib 19.20.0

各ライブラリは上記以外のバージョンでも動作する場合があります。

Each library may work with versions other than those listed above.

# Installation
>## Running with Miniconda
>**Environmental Architecture**   
>Do the following in order.
~~~  bash
$ conda create -n ENVIROMENT-NAME python=3.5
$ conda activate ENVIROMENT-NAME
$ pip install -r REQUIREMENTS.txt
~~~


# Usage
>Basic Usage Introduction
>>**Download the necessary folders.**
~~~ bash
$ git clone https://github.com/remi0628/Express_emotions.git
~~~
>>**Run the file.**
~~~ bash
$ cd Express_emotions/src
$ python main.py
~~~
Let's see your face on the camera.

# Note
OBS Studio、Dlibのインストール方法については他サイトを参考にしてください。

# Author
* Author
  - Chihiro Shimane
  - Itsuki Okada
* Belonging
  - Tokyo University of Technology Department of Media
* E-mail
  - m011813471@edu.teu.ac.jp[Shimane]
  - m011906297@edu.teu.ac.jp[Okada]
# The underlying program
[Face classification and detection.](https://github.com/oarriaga/face_classification)
>Real-time face detection and emotion/gender classification using fer2013/IMDB datasets with a keras CNN model and openCV.  
* IMDB gender classification test accuracy: 96%.  
* fer2013 emotion classification test accuracy: 66%.

[OpenCVで画像上に別の画像を描画する](https://note.com/npaka/n/nddb33be1b782)
>OpenCVで、背景画像上に透過画像をオーバーレイさせるのは、意外と面倒なので備忘録的に残します。

# License
The image uses a free いらすとや.
Please follow the license on which this program is based.
Places to see. **The underlying program**

# ESTiVision-MP

**ESTiVision-MP** は、Web カメラと MediaPipe を用いて VR 用仮想トラッカーを生成するための GUI アプリケーションです。現在開発中です。

## 特徴

- **カメラキャリブレーション**  
  チェスボード画像（A4 印刷想定）によるカメラパラメータ推定
- **リアルタイム姿勢推定**  
  MediaPipe Pose を用いたキーポイント検出（CPU 動作）
- **GUI アプリ**  
  PySide6 による直感的な操作画面

## ディレクトリ構成・役割

- `devices/`  
  カメラや物理デバイス、キャリブレーション、ストリーム系
- `estimators/`  
  姿勢推定などの AI/ML"推論器"
- `trackers/`  
  推論結果から仮想トラッカーへの変換・平滑化・IK・出力形式の変換
- `gui/`  
  Qt / PySide6 による画面・ウィジェット・UI 全般
- `__main__.py`  
  アプリ起動エントリポイント

## インストール

### 依存パッケージ

- Python 3.11
- mediapipe
- numpy
- opencv-python
- PySide6
- python-osc
- qdarkstyle

### セットアップ

```sh
pip install -e .
```

## 起動方法

### Python モジュールとして実行

```sh
python -m estv
```

### コマンドラインから（インストール済みなら）

```sh
estivision
```

## 補足

- カメラキャリブレーション用のチェスボード画像は `images/`ディレクトリにあります（印刷時は PDF 推奨）。

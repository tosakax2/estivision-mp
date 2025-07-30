from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm


def mm_to_px(mm: float, dpi: int) -> int:
    """mm → px 変換（dpi 指定）。四捨五入で整数化。"""
    return int(round(mm / 25.4 * dpi))


def main(
    *,
    inner_cols: int = 6,                 # 内側コーナー数（横方向）
    inner_rows: int = 9,                 # 内側コーナー数（縦方向）
    square_size_mm: float = 20.0,        # 1 マスの一辺 (mm)
    dpi: int = 300,                      # 印刷解像度
    portrait: bool = True,              # True=縦向き, False=横向き
    out_path: Path = Path("images/chessboard_A4_6x9.png"),
    out_pdf: Path = Path("images/chessboard_A4_6x9.pdf"),
) -> tuple[int, int]:
    """A4 サイズぴったりのキャンバス上にチェスボードを描画し PNG＆PDF保存。戻り値は (幅px, 高さpx)。"""

    # --- A4 キャンバスサイズ計算
    a4_w_mm, a4_h_mm = (210.0, 297.0) if portrait else (297.0, 210.0)
    canvas_w: int = mm_to_px(a4_w_mm, dpi)
    canvas_h: int = mm_to_px(a4_h_mm, dpi)

    # --- チェスボード本体サイズ計算
    squares_x: int = inner_cols + 1
    squares_y: int = inner_rows + 1
    sq_px: int = mm_to_px(square_size_mm, dpi)

    board_w: int = squares_x * sq_px
    board_h: int = squares_y * sq_px

    # --- サイズ検証
    if board_w > canvas_w or board_h > canvas_h:
        raise ValueError(
            f"指定の square_size_mm={square_size_mm} では "
            f"チェスボード({board_w}×{board_h}px) が A4({canvas_w}×{canvas_h}px) に収まりません。"
            " マスを小さくするか横向き(A3 等)を検討してください。"
        )

    # --- 余白 (オフセット) 自動計算
    offset_x: int = (canvas_w - board_w) // 2  # 左余白
    offset_y: int = (canvas_h - board_h) // 2  # 上余白

    # --- キャンバス初期化 (白)
    canvas_img: np.ndarray = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255

    # --- 黒マス描画
    rows = np.arange(board_h) // sq_px
    cols = np.arange(board_w) // sq_px
    checkerboard = (rows[:, None] + cols) % 2 == 0  # True で黒マス
    board_img = np.where(checkerboard, 0, 255).astype(np.uint8)
    canvas_img[offset_y : offset_y + board_h, offset_x : offset_x + board_w] = board_img

    # --- 保存先ディレクトリ作成
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- PNG保存
    Image.fromarray(canvas_img).save(out_path, format="PNG", compress_level=0, dpi=(dpi, dpi))
    print(f"保存完了: '{out_path}' ({canvas_w}×{canvas_h}px @{dpi}dpi)")

    # --- PDF保存
    # Pillowで一時保存したPNG画像をreportlabでA4に等倍貼り付け
    c = canvas.Canvas(str(out_pdf), pagesize=(a4_w_mm * mm, a4_h_mm * mm))
    # reportlabのA4原点は左下、Pillowは左上なので注意
    # drawImage(x, y, width, height, ...)
    c.drawImage(
        str(out_path),
        0, 0,  # x, y（左下基準で0,0）
        a4_w_mm * mm,
        a4_h_mm * mm,
        preserveAspectRatio=False,
        mask='auto'
    )
    c.showPage()
    c.save()
    print(f"保存完了: '{out_pdf}' ({a4_w_mm}mm×{a4_h_mm}mm)")

    return canvas_w, canvas_h


if __name__ == "__main__":
    main()

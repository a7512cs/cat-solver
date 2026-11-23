# 找貓貓解題器

上傳遊戲截圖，網頁會辨識盤面並一步一步顯示解題過程（放貓位置＋推理說明）。

- 網頁版：`index.html` + `solver.js`（純前端，圖片不會上傳）
- 命令列版：`python cat-solver.py 截圖.jpg [輸出圖.png]`（需要 numpy、opencv-python、scipy、Pillow）

規則：每一列、每一欄、每種顏色各一隻貓；貓不能相鄰（含斜角）。

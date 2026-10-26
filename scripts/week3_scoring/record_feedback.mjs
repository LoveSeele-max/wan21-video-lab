import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const dir = "E:/cstudy/AIGC/V/outputs/week3-scoring-20261018";
const target = `${dir}/week3_scoring.xlsx`;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(target));
const before = await workbook.render({
  sheetName: "单视频评分", range: "A7:G9", scale: 1.2, format: "png",
});
await fs.writeFile(`${dir}/feedback-before.png`, new Uint8Array(await before.arrayBuffer()));
console.log((await workbook.inspect({
  kind: "table", range: "'单视频评分'!A7:G9", include: "values,formulas",
  tableMaxRows: 3, tableMaxCols: 7, maxChars: 2400,
})).ndjson);
const sheet = workbook.worksheets.getItem("单视频评分");
sheet.getRange("G9").values = [[
  "用户观察：没有人物主体，只见单手和四指；遮挡或少指未确认。基本定住，没有明显挥手。用户保留运动1分，属校准主观评分，不作正式判档锚点。",
]];
sheet.getRange("A2").values = [["视频人工评分 v1.0"]];
sheet.getRange("A3").values = [["标准冻结于2026-10-18；三项分别记录，不求总分。C01/C02仅作校准，保留用户原分。"]];
workbook.worksheets.getItem("配对评分").getRange("A2").values = [["视频配对评分 v1.0"]];
const guide = workbook.worksheets.getItem("评分说明");
guide.getRange("A2").values = [["评分说明 v1.0"]];
guide.getRange("A3").values = [["标准冻结于2026-10-18；校准样例保留原分，不纳入正式盲评。"]];
guide.getRange("B17").values = [["C01/C02为已知身份校准，保留主观原分，不作正式档位锚点；V/AB为待分配编号。"]];
workbook.recalculate();
const after = await workbook.render({
  sheetName: "单视频评分", range: "A7:G9", scale: 1.2, format: "png",
});
await fs.writeFile(`${dir}/feedback-after.png`, new Uint8Array(await after.arrayBuffer()));
for (const name of ["单视频评分", "配对评分", "评分说明"]) {
  const preview = await workbook.render({
    sheetName: name, range: name === "评分说明" ? "A1:E8" : "A1:G9",
    scale: 1, format: "png",
  });
  await fs.writeFile(`${dir}/${name}-v1.png`, new Uint8Array(await preview.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(workbook)).save(`${dir}/feedback-updated.xlsx`);
console.log("Staged update: feedback-updated.xlsx");

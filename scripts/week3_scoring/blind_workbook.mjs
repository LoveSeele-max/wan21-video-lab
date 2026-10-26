import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = "E:/cstudy/AIGC/V";
const name = process.argv[2] || "week3-R01-20261021";
const dir = path.join(root, "outputs", name);
const tasks = JSON.parse(await fs.readFile(path.join(dir, "tasks.json"), "utf8"));
const destination = path.join(dir, "评分表.xlsx");
try { await fs.access(destination); throw new Error("Do not overwrite existing ratings"); }
catch (e) { if (e.code !== "ENOENT") throw e; }
const wb = Workbook.create();
const single = wb.worksheets.add("单视频评分");
const pair = wb.worksheets.add("配对评分");
const session = wb.worksheets.add("会话记录");
const scores = ["2-较好", "1-有缺陷", "0-明显失败", "无法判断"];
const directions = ["A更好", "无明确优劣", "B更好", "无法判断"];
const severity = ["无明显差异", "有差异但难判优劣", "轻微", "明显", "无法判断"];
const state = ["可播放", "无法播放"];
const view = ["原速", "原速+暂停", "原速+慢放", "原速+逐帧"];
function list(sheet, range, values) {
  sheet.getRange(range).dataValidation = { rule: { type: "list", values } };
}
function base(sheet, last, end, widths, title, note) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${last}${end}`).format.font = {
    name: "Microsoft YaHei", size: 11, color: "#202B35",
  };
  sheet.getRange(`A1:${last}${end}`).format.verticalAlignment = "center";
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { size: 16, bold: true };
  sheet.getRange(`A2:${last}2`).format.rowHeight = 32;
  sheet.getRange("A3").values = [[note]];
  sheet.getRange("A3").format.font = { color: "#5C6670", size: 10 };
  widths.forEach((width, col) => {
    sheet.getRangeByIndexes(0, col, end, 1).format.columnWidth = width;
  });
}
function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: "#365563", font: { name: "Microsoft YaHei", size: 11, bold: true, color: "#FFFFFF" },
    wrapText: true, horizontalAlignment: "center", verticalAlignment: "center", rowHeight: 40,
  };
}
base(single, "L", 23, [12, 16, 17, 17, 13, 17, 43, 19, 16, 51, 49, 27],
  "单视频独立评分 · R01", "按编号依次观看；黄色为填写区；完整规则见评分标准.txt。");
single.getRange("A5:F5").values = [["结构已评", null, "时序已评", null, "符合度已评", null]];
single.getRange("B5").formulas = [["=COUNTA(B8:B23)"]];
single.getRange("D5").formulas = [["=COUNTA(C8:C23)"]];
single.getRange("F5").formulas = [["=COUNTA(D8:D23)"]];
single.getRange("A7:L7").values = [[
  "视频编号", "画质与结构", "时序与运动自然度", "提示词符合度", "观看状态",
  "问题时间（秒）", "观察证据／无法判断原因", "查看方式", "评分耗时（秒）",
  "提示词", "动作要求", "视频位置",
]];
single.getRange("A8:L23").values = tasks.singles.map(t => [
  t.id, null, null, null, null, null, null, null, null, t.prompt, t.requirement, t.video,
]);
single.getRange("B8:I23").format.fill = "#FFF5D8";
single.getRange("A8:L23").format.rowHeight = 110;
single.getRange("F8:G23").format.wrapText = true;
single.getRange("J8:L23").format.wrapText = true;
single.getRange("I8:I23").setNumberFormat("0");
list(single, "B8:D23", scores); list(single, "E8:E23", state); list(single, "H8:H23", view);
single.dataValidations.add({ range: "I8:I23", rule: { type: "whole", operator: "between", formula1: 0, formula2: 86400 } });
single.tables.add("A7:L23", true, "SingleReview");
header(single, "A7:L7");
single.freezePanes.freezeRows(7); single.freezePanes.freezeColumns(1);

base(pair, "O", 11, [12, 17, 23, 17, 23, 17, 23, 13, 17, 43, 16, 19, 51, 49, 27],
  "配对独立评分 · R01", "完成单视频阶段后再评；左侧 A，右侧 B；三项分别比较。");
pair.getRange("A5:F5").values = [["结构已评", null, "时序已评", null, "符合度已评", null]];
pair.getRange("B5").formulas = [["=COUNTA(B8:B11)"]];
pair.getRange("D5").formulas = [["=COUNTA(D8:D11)"]];
pair.getRange("F5").formulas = [["=COUNTA(F8:F11)"]];
pair.getRange("A7:O7").values = [[
  "配对编号", "结构方向", "结构程度", "时序方向", "时序程度", "符合度方向", "符合度程度",
  "观看状态", "问题时间（秒）", "观察证据／无法判断原因", "评分耗时（秒）", "查看方式",
  "提示词", "动作要求", "视频位置",
]];
pair.getRange("A8:O11").values = tasks.pairs.map(t => [
  t.id, null, null, null, null, null, null, null, null, null, null, null, t.prompt, t.requirement, t.video,
]);
pair.getRange("B8:L11").format.fill = "#FFF5D8";
pair.getRange("A8:O11").format.rowHeight = 120;
pair.getRange("I8:J11").format.wrapText = true;
pair.getRange("M8:O11").format.wrapText = true;
for (const col of ["B", "D", "F"]) list(pair, `${col}8:${col}11`, directions);
for (const col of ["C", "E", "G"]) list(pair, `${col}8:${col}11`, severity);
list(pair, "H8:H11", state); list(pair, "L8:L11", view);
pair.getRange("K8:K11").setNumberFormat("0");
pair.dataValidations.add({ range: "K8:K11", rule: { type: "whole", operator: "between", formula1: 0, formula2: 86400 } });
pair.tables.add("A7:O11", true, "PairReview");
header(pair, "A7:O7");
pair.freezePanes.freezeRows(7); pair.freezePanes.freezeColumns(1);

base(session, "C", 24, [32, 27, 83], "评分会话记录", "日期按实际填写；总用时不含读标准、准备及休息。");
session.getRange("A5:C19").values = [
  ["记录项", "填写／计算结果", "说明"],
  ["评分者编号", "R01", "仅用编号，不需要姓名"],
  ["评分日期", null, "填写实际日期，例如 2026-10-21"],
  ["播放器", null, "同一播放器、原速、相近显示大小"],
  ["单视频阶段总用时（秒）", null, "播放、重看和填表总用时，排除读标准、准备和休息"],
  ["配对阶段总用时（秒）", null, "独立计时，不并入单视频阶段"],
  ["单视频阶段休息（秒）", null, "如有休息，单独记录；无休息填 0"],
  ["配对阶段休息（秒）", null, "如有休息，单独记录；无休息填 0"],
  ["单视频逐条用时合计（秒）", null, "所有 16 条均填用时后显示"],
  ["配对逐条用时合计（秒）", null, "所有 4 对均填用时后显示"],
  ["单视频每条均时（秒）", null, "阶段总用时除以 16；需完成全部三项评分或记录无法播放"],
  ["配对每对均时（秒）", null, "阶段总用时除以 4；需完成三项方向、程度或记录无法播放"],
  ["评分前是否看过待评视频", null, "如是，写明范围；不要查阅身份映射"],
  ["是否与他人讨论评分", null, "记录讨论时间和范围；独立评分后可讨论"],
  ["疑问与犹豫点", null, "保留原始答案，不为迎合他人修改"],
];
header(session, "A5:C5");
session.getRange("A6:C19").format.rowHeight = 40;
session.getRange("A6:C19").format.wrapText = true;
session.getRange("B7:B12").format.fill = "#FFF5D8";
session.getRange("B17:B19").format.fill = "#FFF5D8";
session.getRange("B7").setNumberFormat("yyyy-mm-dd");
session.getRange("B9:B16").setNumberFormat("0");
session.getRange("B13").formulas = [[`=IF(COUNT('单视频评分'!I8:I23)=16,SUM('单视频评分'!I8:I23),"")`]];
session.getRange("B14").formulas = [[`=IF(COUNT('配对评分'!K8:K11)=4,SUM('配对评分'!K8:K11),"")`]];
const completeSingle = `AND(COUNTA('单视频评分'!E8:E23)=16,(COUNTA('单视频评分'!B8:D23)+3*COUNTIF('单视频评分'!E8:E23,"无法播放"))=48)`;
const completePair = `AND(COUNTA('配对评分'!H8:H11)=4,(COUNTA('配对评分'!B8:G11)+6*COUNTIF('配对评分'!H8:H11,"无法播放"))=24)`;
session.getRange("B15").formulas = [[`=IF(AND(ISNUMBER(B9),${completeSingle}),B9/16,"")`]];
session.getRange("B16").formulas = [[`=IF(AND(ISNUMBER(B10),${completePair}),B10/4,"")`]];
for (const cell of ["B9", "B10", "B11", "B12"]) {
  session.dataValidations.add({ range: cell, rule: { type: "whole", operator: "between", formula1: 0, formula2: 86400 } });
}
// Check formula prerequisites using temporary values, then restore all input blanks.
single.getRange("B8:D23").values = [["2-较好"]];
single.getRange("E8:E23").values = [["可播放"]];
session.getRange("B9").values = [[160]];
wb.recalculate();
if (single.getRange("B5").values[0][0] !== 16 ||
    session.getRange("B15").values[0][0] !== 10) {
  console.log((await wb.inspect({ kind: "table", range: "'会话记录'!A9:C16",
    include: "values,formulas", maxChars: 2500, tableMaxRows: 8 })).ndjson);
  console.log("Count:", single.getRange("B5").values);
  throw new Error("Completion/count test failed");
}
single.getRange("B8:E23").clear({ applyTo: "contents" });
session.getRange("B9").clear({ applyTo: "contents" });
wb.recalculate();
if (single.getRange("B5").values[0][0] !== 0 ||
    session.getRange("B15").values[0][0]) throw new Error("Blank template test failed");
const errors = await wb.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!",
  options: { useRegex: true, maxResults: 10 }, maxChars: 1200,
});
console.log(errors.ndjson);
const previewDir = path.join(root, "notes/blind_private", name);
for (const [sheetName, range, file] of [
  ["单视频评分", "A5:G9", "single-sheet.png"],
  ["单视频评分", "H7:L9", "prompt-sheet.png"],
  ["配对评分", "A5:G9", "pair-sheet.png"],
  ["会话记录", "A5:C19", "session-sheet.png"],
]) {
  const image = await wb.render({ sheetName, range, scale: 1.3, format: "png" });
  await fs.writeFile(path.join(previewDir, file), new Uint8Array(await image.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(destination);
console.log("Blank workbook exported.");

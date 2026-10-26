import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = "E:/cstudy/AIGC/V";
const out = path.join(root, "outputs/week3-scoring-20261018");
await fs.mkdir(out, { recursive: true });
const wb = Workbook.create();
const single = wb.worksheets.add("单视频评分");
const pair = wb.worksheets.add("配对评分");
const guide = wb.worksheets.add("评分说明");
const colors = { ink: "#253247", header: "#305763", input: "#FFF8E8", muted: "#667085" };
const scores = ["2-较好", "1-有缺陷", "0-明显失败", "无法判断"];
const directions = ["A更好", "无明确优劣", "B更好", "无法判断"];
const severity = ["无明显差异", "有差异但难判优劣", "轻微", "明显", "无法判断"];

function base(sheet, last, rows) {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${last}${rows}`).format.font = { name: "Microsoft YaHei", size: 11, color: colors.ink };
  sheet.getRange(`A1:${last}${rows}`).format.verticalAlignment = "center";
  sheet.getRange(`A1:${last}${rows}`).format.rowHeight = 28;
  sheet.getRange("A2").format.font = { size: 16, bold: true };
  sheet.getRange(`A2:${last}2`).format.rowHeight = 34;
  sheet.getRange(`A3:${last}3`).format.font = { color: colors.muted, size: 10 };
}
function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: colors.header,
    font: { name: "Microsoft YaHei", color: "#FFFFFF", bold: true, size: 11 },
    horizontalAlignment: "center", verticalAlignment: "center", wrapText: true, rowHeight: 40,
  };
}
function list(sheet, range, values) {
  sheet.getRange(range).dataValidation = { rule: { type: "list", values } };
}
function metricColors(sheet, range) {
  for (const [text, fill] of [
    ["2-较好", "#E8F3EC"], ["1-有缺陷", "#FFF0CD"],
    ["0-明显失败", "#FBE8E8"], ["无法判断", "#ECEEF2"],
  ]) sheet.getRange(range).conditionalFormats.add("containsText", { text, format: { fill } });
}

base(single, "M", 48);
single.tabColor = colors.header;
single.getRange("A2").values = [["视频人工评分 v0.1"]];
single.getRange("A3").values = [["草案待试评；三项分别记录，不求总分。黄色为填写区。先评 C01、C02。"]];
single.getRange("A5:F5").values = [["结构已评", null, "时序已评", null, "符合度已评", null]];
single.getRange("B5").formulas = [["=COUNTA(B8:B47)"]];
single.getRange("D5").formulas = [["=COUNTA(C8:C47)"]];
single.getRange("F5").formulas = [["=COUNTA(D8:D47)"]];
single.getRange("H5").values = [["已评数量含“无法判断”，不是合格数。"]];
single.getRange("A7:M7").values = [[
  "视频编号", "画质与结构", "时序与运动", "提示词符合度", "观看状态",
  "问题时间（秒）", "观察证据／无法判断原因", "提示词与动作要求", "视频位置",
  "查看方式", "评分耗时（秒）", "评分者", "评分日期",
]];
const singleRows = Array.from({ length: 40 }, (_, i) => [
  i < 2 ? `C0${i + 1}` : `V${String(i - 1).padStart(3, "0")}`,
  null, null, null, null, null, null, null, null, null, null, null, null,
]);
const expectation = "A person waving one hand, plain background, static camera. 人物单手挥手，至少一次可辨认往返。";
singleRows[0][7] = expectation;
singleRows[0][8] = path.join(root, "video/20261013-165316-160_42_.mp4");
singleRows[1][7] = expectation;
singleRows[1][8] = path.join(root, "local/outputs/week2/L0-stages-20261017-171311-3d8ca2f4/run-01/20261017-172339-409_42_.mp4");
single.getRange("A8:M47").values = singleRows;
single.getRange("B8:M47").format.fill = colors.input;
single.getRange("F8:I47").format.wrapText = true;
single.getRange("A8:M47").format.rowHeight = 62;
single.getRange("K8:K47").setNumberFormat("0");
single.getRange("M8:M47").setNumberFormat("yyyy-mm-dd");
list(single, "B8:D47", scores);
list(single, "E8:E47", ["待看", "可播放", "OOM", "生成失败", "无法播放"]);
list(single, "J8:J47", ["原速", "原速+暂停", "原速+慢放", "原速+逐帧"]);
metricColors(single, "B8:D47");
single.dataValidations.add({ range: "K8:K47", rule: { type: "whole", operator: "between", formula1: 0, formula2: 86400 } });
const singleWidths = [12, 15, 16, 16, 14, 18, 48, 58, 58, 18, 18, 14, 17];
singleWidths.forEach((w, i) => { single.getRangeByIndexes(0, i, 48, 1).format.columnWidth = w; });
single.tables.add("A7:M47", true, "SingleRatings");
header(single, "A7:M7");
single.freezePanes.freezeRows(7);
single.freezePanes.freezeColumns(1);

base(pair, "P", 48);
pair.tabColor = "#60868F";
pair.getRange("A2").values = [["视频配对评分 v0.1"]];
pair.getRange("A3").values = [["A/B 身份单独保存；各维度分别判断方向和程度，不做总胜负。"]];
pair.getRange("A5:D5").values = [["结构已评", null, "运动已评", null]];
pair.getRange("B5").formulas = [["=COUNTA(B8:B47)"]];
pair.getRange("D5").formulas = [["=COUNTA(D8:D47)"]];
pair.getRange("F5:G5").values = [["符合度已评", null]];
pair.getRange("G5").formulas = [["=COUNTA(F8:F47)"]];
pair.getRange("A7:P7").values = [[
  "配对编号", "结构方向", "结构程度", "运动方向", "运动程度", "符合度方向", "符合度程度",
  "问题时间（秒）", "观察证据／无法判断原因", "提示词与动作要求", "A/B 视频位置",
  "观看状态", "查看方式", "评分耗时（秒）", "评分者", "评分日期",
]];
pair.getRange("A8:P47").values = Array.from({ length: 40 }, (_, i) => [
  `AB${String(i + 1).padStart(3, "0")}`, ...Array(15).fill(null),
]);
pair.getRange("B8:P47").format.fill = colors.input;
pair.getRange("H8:K47").format.wrapText = true;
pair.getRange("A8:P47").format.rowHeight = 56;
for (const col of ["B", "D", "F"]) list(pair, `${col}8:${col}47`, directions);
for (const col of ["C", "E", "G"]) list(pair, `${col}8:${col}47`, severity);
list(pair, "L8:L47", ["待看", "双方可播放", "A失败或无法播放", "B失败或无法播放", "双方失败或无法播放"]);
list(pair, "M8:M47", ["原速", "原速+暂停", "原速+慢放", "原速+逐帧"]);
pair.getRange("N8:N47").setNumberFormat("0");
pair.getRange("P8:P47").setNumberFormat("yyyy-mm-dd");
pair.dataValidations.add({ range: "N8:N47", rule: { type: "whole", operator: "between", formula1: 0, formula2: 86400 } });
const pairWidths = [12, 17, 23, 17, 23, 17, 23, 18, 48, 58, 58, 24, 18, 18, 14, 17];
pairWidths.forEach((w, i) => { pair.getRangeByIndexes(0, i, 48, 1).format.columnWidth = w; });
pair.tables.add("A7:P47", true, "PairRatings");
header(pair, "A7:P7");
pair.freezePanes.freezeRows(7);
pair.freezePanes.freezeColumns(1);

base(guide, "E", 40);
guide.tabColor = "#87919C";
guide.getRange("A2").values = [["评分说明 v0.1"]];
guide.getRange("A3").values = [["待 C01、C02 试评后冻结；本表没有任何预填评分。"]];
guide.getRange("A5:E8").values = [
  ["维度", "2-较好", "1-有缺陷", "0-明显失败", "无法判断"],
  ["画质与结构", "主体清楚，形状合理，无明显结构缺陷", "局部模糊、融合或轻度变形，仍可辨认", "严重扭曲、多肢体或模糊，破坏主体与动作识别", "该维度关键内容不可见，写明原因"],
  ["时序与运动", "变化连贯，身份稳定，关节运动合理", "偶发闪烁、抖动或不自然运动，过程仍可理解", "明显跳变、换人、肢体瞬移或反复异常运动", "关键关节被遮挡等，无法观察"],
  ["提示词符合度", "主体、动作及明确场景和镜头要求基本满足", "动作部分完成，或遗漏部分场景和镜头条件", "核心主体或动作缺失、错误", "缺提示词或严重遮挡使动作无法辨认"],
];
header(guide, "A5:E5");
guide.getRange("A6:E8").format.wrapText = true;
guide.getRange("A6:E8").format.rowHeight = 92;
guide.getRange("A10:B17").values = [
  ["记录规则", "如何填写"],
  ["空白", "尚未评，不表示 0 分。0/1/无法判断要附时间和证据。"],
  ["观看", "原速完整看两遍；必要时辅助查看并记录；仅逐帧可见的小问题不直接判明显失败。"],
  ["动作缺失", "清楚看到没做要求的动作，符合度为 0；不自动把时序也判 0。"],
  ["失败样本", "OOM、生成失败或无法播放：填观看状态，分数留空，证据写原因。"],
  ["配对方向", "三项各选 A更好/无明确优劣/B更好/无法判断；不填配置身份。"],
  ["配对程度", "无明确优劣配无明显差异/有差异但难判优劣；A或B更好配轻微/明显；无法判断配无法判断。"],
  ["试评与盲评", "C01/C02 是已知身份校准样例，不计正式盲评；V/AB 为待分配的空白编号。"],
];
header(guide, "A10:B10");
guide.getRange("A11:B17").format.wrapText = true;
guide.getRange("A11:B17").format.rowHeight = 68;
guide.getRange("A19:C31").values = [
  ["提示词草案", "核心事件", "验收要点（最终提示词冻结时复核）"],
  ["P01", "挥手", "人物单手至少一次可辨认往返"],
  ["P02", "点头", "头部向下再返回，至少一次"],
  ["P03", "转身", "从非正面通过转身变成面向镜头"],
  ["P04", "行走", "交替步态与向右位移，全身可见"],
  ["P05", "原地慢跑", "交替跑步动作，整体位置基本保持"],
  ["P06", "拍手", "双手靠近、合拢、再分开，至少一次"],
  ["P07", "海浪", "海浪推进并拍打岩石"],
  ["P08", "烛火", "火焰自身摇曳，非全画面闪烁"],
  ["P09", "航拍", "雪山镜头连续缓慢横移"],
  ["P10", "奔跑", "犬有交替跑动与跨草地位移"],
  ["P11", "街景", "夜间城市中至少一辆车相对街道经过"],
  ["P12", "蒸汽", "蒸汽相对杯口向上运动"],
];
header(guide, "A19:C19");
guide.getRange("A20:C31").format.wrapText = true;
guide.getRange("A20:C31").format.rowHeight = 44;
[20, 58, 56, 56, 34].forEach((w, i) => { guide.getRangeByIndexes(0, i, 40, 1).format.columnWidth = w; });
guide.getRange("A33").values = [["完整规则：项目 notes/week3_rubric.md；来源为 METRICS、BASELINE_PROBE 与 Week 3 计划。"]];

// Exercise editable fields and the linked counts, then restore the empty template.
single.getRange("B8:D8").values = [["0-明显失败", "无法判断", "2-较好"]];
wb.recalculate();
if (single.getRange("B5").values[0][0] !== 1 ||
    single.getRange("D5").values[0][0] !== 1 ||
    single.getRange("F5").values[0][0] !== 1) throw new Error("Single rating counts did not update.");
pair.getRange("B8").values = [["A更好"]];
wb.recalculate();
if (pair.getRange("B5").values[0][0] !== 1) throw new Error("Pair count did not update.");
single.getRange("B8:D8").clear({ applyTo: "contents" });
pair.getRange("B8").clear({ applyTo: "contents" });
wb.recalculate();
if (single.getRange("B5").values[0][0] !== 0 || pair.getRange("B5").values[0][0] !== 0) throw new Error("Template was not reset.");
console.log((await wb.inspect({ kind: "table", range: "'单视频评分'!A5:G9", include: "values,formulas", tableMaxRows: 5, tableMaxCols: 7, maxChars: 1600 })).ndjson);
const errors = await wb.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 20 }, maxChars: 1500,
});
console.log(errors.ndjson);
for (const [sheetName, range, file] of [
  ["单视频评分", "A1:G11", "single-preview.png"],
  ["配对评分", "A1:G11", "pair-preview.png"],
  ["评分说明", "A1:E8", "guide-preview.png"],
]) {
  const image = await wb.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(path.join(out, file), new Uint8Array(await image.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(path.join(out, "week3_scoring.xlsx"));
console.log("Exported:", path.join(out, "week3_scoring.xlsx"));

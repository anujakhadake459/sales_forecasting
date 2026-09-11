let DATA = [];
let FEATURES = [];
let MODEL = null;
let forecastType = "product";

const $ = id => document.getElementById(id);

const productInputs = $("productInputs");
const storeInputs = $("storeInputs");

document.querySelectorAll(".seg").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg").forEach(x => x.classList.remove("active"));
    btn.classList.add("active");
    forecastType = btn.dataset.type;
    productInputs.classList.toggle("hidden", forecastType !== "product");
    storeInputs.classList.toggle("hidden", forecastType !== "store");
    updateInfo();
  });
});

function unique(values) {
  return [...new Set(values.filter(v => v !== null && v !== undefined && v !== ""))];
}
function sortText(a,b){return String(a).localeCompare(String(b));}
function sortNum(a,b){return Number(a)-Number(b);}

function fillSelect(id, values, formatter=x=>x) {
  const el = $(id);
  el.innerHTML = "";
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = String(v);
    opt.textContent = formatter(v);
    el.appendChild(opt);
  });
}

function setValue(id, value) {
  if (value === null || value === undefined) return;
  $(id).value = String(value);
}

function pairRows(product, store) {
  return DATA.filter(r => r.Product_ID === product && r.Store_ID === store)
             .sort((a,b)=>a.Date.localeCompare(b.Date));
}

function storeRows(store) {
  return DATA.filter(r => r.Store_ID === store)
             .sort((a,b)=>a.Date.localeCompare(b.Date));
}

function productName(id) {
  const r = DATA.find(x=>x.Product_ID===id);
  return r ? String(r.Product_Name) : id;
}
function storeName(id) {
  const r = DATA.find(x=>x.Store_ID===id);
  return r ? String(r.Store_Location) : id;
}
function categoryFor(product, store) {
  const rows = pairRows(product, store);
  return rows.length ? rows[rows.length-1].Category : "";
}
function getSeason(month){
  if ([12,1,2].includes(month)) return "Winter";
  if ([3,4,5].includes(month)) return "Spring";
  if ([6,7,8].includes(month)) return "Summer";
  return "Autumn";
}
function dateParts(dateStr){
  const d = new Date(dateStr + "T00:00:00");
  const days=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
  const months=["January","February","March","April","May","June","July","August","September","October","November","December"];
  const month=d.getMonth()+1;
  return {
    day:d.getDate(),
    month,
    dayOfWeek:days[d.getDay()],
    monthName:months[d.getMonth()],
    quarter:"Q"+Math.floor((month-1)/3+1),
    isWeekend:(d.getDay()===0 || d.getDay()===6) ? 1 : 0,
    season:getSeason(month)
  };
}
function addDays(dateStr, n){
  const d = new Date(dateStr + "T00:00:00");
  d.setDate(d.getDate()+n);
  return d.toISOString().slice(0,10);
}
function todayIST(){
  // Browser's local clock is normally enough for the UI; use Intl to obtain India date.
  return new Intl.DateTimeFormat("en-CA",{timeZone:"Asia/Kolkata",year:"numeric",month:"2-digit",day:"2-digit"}).format(new Date());
}
function holidayFlag(dateStr){
  const p = dateParts(dateStr);
  return DATA.some(r => Number(r.Holiday_Flag) === 1 &&
                       new Date(r.Date+"T00:00:00").getMonth()+1 === p.month &&
                       new Date(r.Date+"T00:00:00").getDate() === p.day) ? 1 : 0;
}
function datasetTypeValue(column, displayValue){
  const sample = DATA.find(r=>r[column]!==null && r[column]!==undefined);
  if (!sample) return displayValue;
  const v=sample[column];
  if (typeof v === "number") {
    if (String(displayValue).toLowerCase()==="yes") return 1;
    if (String(displayValue).toLowerCase()==="no") return 0;
  }
  return displayValue;
}
function optionIndex(values, val){
  const i=values.map(String).indexOf(String(val));
  return i<0?0:i;
}

function initializeUI(){
  const regions = unique(DATA.map(r=>r.Store_Location)).sort(sortText);
  fillSelect("region", regions);
  const stores = unique(DATA.map(r=>r.Store_ID)).sort(sortText);
  fillSelect("storeTotal", stores, v=>`${v} - ${storeName(v)}`);

  fillSelect("product", unique(DATA.map(r=>r.Product_ID)).sort(sortText),
             v=>`${v} - ${productName(v)}`);

  fillSelect("promotion", unique(DATA.map(r=>r.Promotion_Flag)).sort(sortText));
  fillSelect("localEvent", unique(DATA.map(r=>r.Local_Event_Flag)).sort(sortText));
  fillSelect("weather", unique(DATA.map(r=>String(r.Weather))).sort(sortText));
  fillSelect("channel", unique(DATA.map(r=>String(r.Sales_Channel))).sort(sortText));
  fillSelect("segment", unique(DATA.map(r=>String(r.Customer_Segment))).sort(sortText));

  $("region").addEventListener("change", updateStoresForRegion);
  $("store").addEventListener("change", updateBusinessDefaults);
  $("product").addEventListener("change", updateBusinessDefaults);
  $("predictBtn").addEventListener("click", runForecast);

  updateStoresForRegion();

  const latest = DATA.map(r=>r.Date).sort().at(-1);
  const minDate = [todayIST(), addDays(latest,1)].sort().at(-1);
  $("startDate").min = addDays(latest,1);
  $("startDate").value = minDate;
  updateInfo();
}

function updateStoresForRegion(){
  const region = $("region").value;
  const rows = DATA.filter(r=>String(r.Store_Location)===String(region));
  const ids = unique(rows.map(r=>r.Store_ID)).sort(sortText);
  fillSelect("store", ids, v=>`${v} - ${storeName(v)}`);
  updateBusinessDefaults();
}
function updateBusinessDefaults(){
  const product=$("product").value, store=$("store").value;
  let rows=pairRows(product,store);
  if(!rows.length) rows=storeRows(store);
  if(!rows.length) return;
  const r=rows.at(-1);
  setValue("price", r.Price);
  setValue("discount", r.Discount_Percentage);
  setValue("promotion", r.Promotion_Flag);
  setValue("stock", r.Stock_Availability);
  setValue("localEvent", r.Local_Event_Flag);
  setValue("competitor", r.Competitor_Price);
  setValue("economic", r.Economic_Indicator);
  setValue("marketing", r.Marketing_Spend);
  setValue("weather", r.Weather);
  setValue("channel", r.Sales_Channel);
  setValue("segment", r.Customer_Segment);
}
function updateInfo(){
  $("infoMessage").textContent = forecastType==="product"
    ? "Forecast starts on the selected date. Region filters the available store; Product + Store and business features are used for the forecast."
    : "Store Total forecasts every eligible product in the selected store and sums their daily predictions.";
}

function buildVector(product,store,dateStr,history,userInputs){
  const h=history.filter(r=>r.Product_ID===product && r.Store_ID===store)
                 .sort((a,b)=>a.Date.localeCompare(b.Date));
  if(h.length<30) throw new Error(`${product} + ${store} has only ${h.length} historical observations. At least 30 are required.`);

  const sales=h.map(r=>Number(r.Units_Sold));
  const latest=h.at(-1);
  const p=dateParts(dateStr);
  const holidayUser=userInputs.Holiday_Flag;
  const holiday = holidayUser===undefined ? holidayFlag(dateStr) : holidayUser;

  const raw = {
    Product_ID:product,
    Category:latest.Category,
    Store_ID:store,
    Price:userInputs.Price ?? Number(latest.Price),
    Discount_Percentage:userInputs.Discount_Percentage ?? Number(latest.Discount_Percentage),
    Promotion_Flag:userInputs.Promotion_Flag ?? latest.Promotion_Flag,
    Stock_Availability:userInputs.Stock_Availability ?? Number(latest.Stock_Availability),
    Day_of_Week:p.dayOfWeek,
    Month:p.monthName,
    Quarter:p.quarter,
    Holiday_Flag:holiday,
    Is_Weekend:p.isWeekend,
    Season:p.season,
    Weather:userInputs.Weather ?? latest.Weather,
    Local_Event_Flag:userInputs.Local_Event_Flag ?? latest.Local_Event_Flag,
    Competitor_Price:userInputs.Competitor_Price ?? Number(latest.Competitor_Price),
    Economic_Indicator:userInputs.Economic_Indicator ?? Number(latest.Economic_Indicator),
    Sales_Channel:userInputs.Sales_Channel ?? latest.Sales_Channel,
    Customer_Segment:userInputs.Customer_Segment ?? latest.Customer_Segment,
    Marketing_Spend:userInputs.Marketing_Spend ?? Number(latest.Marketing_Spend),
    Lag_1:sales.at(-1),
    Lag_7:sales.at(-7),
    Lag_14:sales.at(-14),
    Rolling_Mean_7:sales.slice(-7).reduce((a,b)=>a+b,0)/7,
    Rolling_Mean_14:sales.slice(-14).reduce((a,b)=>a+b,0)/14,
    Rolling_Mean_30:sales.slice(-30).reduce((a,b)=>a+b,0)/30
  };

  const x = new Array(FEATURES.length).fill(0);
  const numeric = new Set([
    "Price","Discount_Percentage","Promotion_Flag","Stock_Availability",
    "Holiday_Flag","Is_Weekend","Local_Event_Flag","Competitor_Price",
    "Economic_Indicator","Marketing_Spend","Lag_1","Lag_7","Lag_14",
    "Rolling_Mean_7","Rolling_Mean_14","Rolling_Mean_30"
  ]);

  for(let i=0;i<FEATURES.length;i++){
    const f=FEATURES[i];
    if(numeric.has(f)){
      x[i]=Number(raw[f] ?? 0);
    } else {
      const underscore=f.indexOf("_");
      const prefix=underscore>=0 ? f.slice(0,underscore) : f;
      // Feature names are prefix + "_" + category value.
      if(f.startsWith("Product_ID_")) x[i]=(raw.Product_ID===f.slice("Product_ID_".length)?1:0);
      else if(f.startsWith("Category_")) x[i]=(String(raw.Category)===f.slice("Category_".length)?1:0);
      else if(f.startsWith("Store_ID_")) x[i]=(raw.Store_ID===f.slice("Store_ID_".length)?1:0);
      else if(f.startsWith("Day_of_Week_")) x[i]=(raw.Day_of_Week===f.slice("Day_of_Week_".length)?1:0);
      else if(f.startsWith("Month_")) x[i]=(raw.Month===f.slice("Month_".length)?1:0);
      else if(f.startsWith("Quarter_")) x[i]=(raw.Quarter===f.slice("Quarter_".length)?1:0);
      else if(f.startsWith("Season_")) x[i]=(raw.Season===f.slice("Season_".length)?1:0);
      else if(f.startsWith("Weather_")) x[i]=(String(raw.Weather)===f.slice("Weather_".length)?1:0);
      else if(f.startsWith("Sales_Channel_")) x[i]=(String(raw.Sales_Channel)===f.slice("Sales_Channel_".length)?1:0);
      else if(f.startsWith("Customer_Segment_")) x[i]=(String(raw.Customer_Segment)===f.slice("Customer_Segment_".length)?1:0);
      else if(f.startsWith("Holiday_Name_")){
        // The supplied Streamlit prediction code does NOT include Holiday_Name
        // in prediction_row. Therefore these saved-model columns are reindexed
        // with fill_value=0, so they must remain zero here as well.
        x[i]=0;
      }
    }
  }
  return x;
}

function traverseTree(tree,x){
  let node=0;
  const left=tree.left_children, right=tree.right_children;
  const splits=tree.split_indices, thresholds=tree.split_conditions;
  const missingLeft=tree.default_left || [];
  while(left[node]!==-1 || right[node]!==-1){
    const f=splits[node];
    const v=x[f];
    if(Number.isNaN(v)){
      node = missingLeft[node] ? left[node] : right[node];
    }else{
      node = (v < thresholds[node]) ? left[node] : right[node];
    }
  }
  return tree.base_weights[node];
}

function predictVector(x){
  const learner=MODEL.learner;
  const base=parseFloat(String(learner.learner_model_param.base_score).replace(/[\[\]]/g,""));
  const trees=learner.gradient_booster.model.trees;
  let prediction=base;
  // XGBoost's exported JSON stores the tree leaf weights after the configured learning-rate scaling.
  for(const tree of trees) prediction += traverseTree(tree,x);
  return Math.max(0,Math.round(prediction));
}

function userInputObject(){
  if(forecastType==="store") return {};
  const holiday=$("holiday").value;
  const obj={
    Price:Number($("price").value),
    Discount_Percentage:Number($("discount").value),
    Promotion_Flag:datasetTypeValue("Promotion_Flag",$("promotion").value),
    Stock_Availability:Number($("stock").value),
    Local_Event_Flag:datasetTypeValue("Local_Event_Flag",$("localEvent").value),
    Competitor_Price:Number($("competitor").value),
    Economic_Indicator:Number($("economic").value),
    Marketing_Spend:Number($("marketing").value),
    Weather:$("weather").value,
    Sales_Channel:$("channel").value,
    Customer_Segment:$("segment").value
  };
  if(holiday==="Yes") obj.Holiday_Flag=1;
  if(holiday==="No") obj.Holiday_Flag=0;
  return obj;
}

function appendPrediction(history,product,store,dateStr,prediction,userInputs){
  const rows=pairRows(product,store);
  const latest={...rows.at(-1)};
  latest.Date=dateStr;
  latest.Units_Sold=prediction;
  const p=dateParts(dateStr);
  latest.Day_of_Week=p.dayOfWeek;
  latest.Month=p.monthName;
  latest.Quarter=p.quarter;
  latest.Holiday_Flag=holidayFlag(dateStr);
  latest.Is_Weekend=p.isWeekend;
  latest.Season=p.season;
  if(userInputs && Object.keys(userInputs).length){
    for(const [k,v] of Object.entries(userInputs)) if(k in latest) latest[k]=v;
  }
  history.push(latest);
}

function generateForecast(product,store,horizon,startDate,baseHistory,userInputs){
  const history=baseHistory.map(x=>({...x}));
  const pair=pairRows(product,store);
  if(!pair.length) throw new Error(`No historical data found for ${product} + ${store}.`);
  const out=[];
  for(let i=0;i<horizon;i++){
    const d=addDays(startDate,i);
    const x=buildVector(product,store,d,history,userInputs);
    const pred=predictVector(x);
    out.push({Date:d,Product_ID:product,Store_ID:store,Predicted_Units_Sold:pred});
    appendPrediction(history,product,store,d,pred,userInputs);
  }
  return out;
}

function generateStoreTotal(store,horizon,startDate){
  const products=unique(storeRows(store).map(r=>r.Product_ID));
  const forecasts=[];
  const skipped=[];
  for(const p of products){
    if(pairRows(p,store).length<30){skipped.push(p);continue;}
    try{
      forecasts.push(...generateForecast(p,store,horizon,startDate,DATA,{}));
    }catch(e){skipped.push(p);}
  }
  const byDate={};
  forecasts.forEach(r=>{byDate[r.Date]=(byDate[r.Date]||0)+r.Predicted_Units_Sold;});
  const totals=Object.entries(byDate).sort().map(([Date,v])=>({Date,Predicted_Units_Sold:Math.round(v)}));
  return {totals,forecasts,skipped};
}

function formatNumber(n){return Number(n).toLocaleString("en-IN");}

function renderTable(tableId,columns,rows){
  const table=$(tableId);
  table.querySelector("thead").innerHTML="<tr>"+columns.map(c=>`<th>${c.label}</th>`).join("")+"</tr>";
  table.querySelector("tbody").innerHTML=rows.map(r=>"<tr>"+columns.map(c=>`<td>${r[c.key] ?? ""}</td>`).join("")+"</tr>").join("");
}

function drawLineChart(canvasId, series, title){
  const canvas=$(canvasId), rect=canvas.getBoundingClientRect();
  const dpr=window.devicePixelRatio||1;
  canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height;
  ctx.clearRect(0,0,w,h);
  const pad={l:48,r:18,t:25,b:38};
  const vals=series.flatMap(s=>s.values);
  const max=Math.max(...vals,1), min=Math.min(...vals,0);
  const xCount=Math.max(...series.map(s=>s.values.length),1);
  const x=i=>pad.l+(i/(Math.max(xCount-1,1)))*(w-pad.l-pad.r);
  const y=v=>h-pad.b-((v-min)/(max-min||1))*(h-pad.t-pad.b);

  ctx.strokeStyle="#e6e9ef";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const yy=pad.t+i*(h-pad.t-pad.b)/4;
    ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();
    const val=max-(max-min)*i/4;
    ctx.fillStyle="#7a8190";ctx.font="11px Segoe UI";ctx.fillText(Math.round(val).toLocaleString(),5,yy+4);
  }
  series.forEach((s,si)=>{
    ctx.strokeStyle=si===0?"#315efb":"#0f8a4b";ctx.lineWidth=2.5;
    ctx.beginPath();
    s.values.forEach((v,i)=>{const px=x(i),py=y(v); if(i===0)ctx.moveTo(px,py);else ctx.lineTo(px,py);});
    ctx.stroke();
  });
  ctx.fillStyle="#596273";ctx.font="11px Segoe UI";
  const labels=series[0].labels;
  const step=Math.max(1,Math.ceil(labels.length/6));
  labels.forEach((lab,i)=>{if(i%step===0)ctx.fillText(lab.slice(5),x(i)-16,h-13)});
  let lx=pad.l;
  series.forEach((s,i)=>{ctx.fillStyle=i===0?"#315efb":"#0f8a4b";ctx.fillRect(lx,8,10,10);ctx.fillStyle="#596273";ctx.fillText(s.name,lx+15,17);lx+=80+s.name.length*5});
}

function renderResults(mode, forecast, latestHistoricalDate, historyRows, storeSkipped=[]){
  $("welcome").classList.add("hidden");
  $("results").classList.remove("hidden");

  const dates=forecast.map(r=>r.Date);
  const vals=forecast.map(r=>r.Predicted_Units_Sold);
  const total=vals.reduce((a,b)=>a+b,0);
  const avg=Math.round(total/vals.length);
  const peak=Math.max(...vals);

  $("latestDate").textContent=latestHistoricalDate;
  $("forecastStart").textContent=dates[0];
  $("forecastEnd").textContent=dates.at(-1);
  $("metricHorizon").textContent=`${forecast.length} Days`;
  $("metricTotal").textContent=`${formatNumber(total)} Units`;
  $("metricAverage").textContent=`${formatNumber(avg)} Units`;
  $("metricPeak").textContent=`${formatNumber(peak)} Units`;

  if(storeSkipped.length){
    $("storeSkippedWrap").classList.remove("hidden");
    $("storeSkippedWrap").textContent=`Some products were skipped because they did not have at least 30 historical observations: ${storeSkipped.join(", ")}.`;
  }else $("storeSkippedWrap").classList.add("hidden");

  const detailRows=forecast.map(r=>({
    Date:r.Date,
    Product:mode==="product"?`${r.Product_ID} - ${productName(r.Product_ID)}`:`${r.Product_ID} - ${productName(r.Product_ID)}`,
    Store:`${r.Store_ID} - ${storeName(r.Store_ID)}`,
    Predicted:`${formatNumber(r.Predicted_Units_Sold)}`
  }));
  renderTable("forecastTable",[
    {key:"Date",label:"Forecast Date"},
    {key:"Product",label:"Product"},
    {key:"Store",label:"Store"},
    {key:"Predicted",label:"Predicted Units Sold"}
  ],detailRows);

  const actual=historyRows.slice(-60);
  const actualVals=actual.map(r=>Number(r.Units_Sold));
  const actualLabels=actual.map(r=>r.Date);
  const forecastLabels=dates;
  drawLineChart("forecastChart",[
    {name:"Forecast",labels:forecastLabels,values:vals}
  ],"Forecast");

  // Build combined labels for actual + forecast
  const labels=[...actualLabels,...forecastLabels];
  const actualSeries=new Array(labels.length).fill(null);
  actualVals.forEach((v,i)=>actualSeries[i]=v);
  const forecastSeries=new Array(labels.length).fill(null);
  vals.forEach((v,i)=>forecastSeries[actualLabels.length+i]=v);
  // Canvas helper expects values; replace nulls with endpoints for line continuity.
  const actualPlot=actualSeries.map((v,i)=>v===null?(i?actualSeries[i-1]:0):v);
  const forecastPlot=forecastSeries.map((v,i)=>v===null?null:v);
  // Custom second chart because null gaps are useful.
  drawActualForecast("actualForecastChart",labels,actualPlot,forecastPlot);

  const histDisplay=historyRows.slice(-10).map(r=>({
    Date:r.Date,
    Product:`${r.Product_ID} - ${productName(r.Product_ID)}`,
    Category:r.Category,
    Store:`${r.Store_ID} - ${storeName(r.Store_ID)}`,
    Units_Sold:r.Units_Sold,
    Price:r.Price,
    Discount_Percentage:r.Discount_Percentage,
    Promotion_Flag:r.Promotion_Flag,
    Stock_Availability:r.Stock_Availability,
    Day_of_Week:r.Day_of_Week,
    Month:r.Month,
    Quarter:r.Quarter,
    Holiday_Flag:r.Holiday_Flag,
    Is_Weekend:r.Is_Weekend,
    Season:r.Season,
    Weather:r.Weather,
    Local_Event_Flag:r.Local_Event_Flag,
    Competitor_Price:r.Competitor_Price,
    Economic_Indicator:r.Economic_Indicator,
    Sales_Channel:r.Sales_Channel,
    Customer_Segment:r.Customer_Segment,
    Marketing_Spend:r.Marketing_Spend
  }));
  $("historySummary").textContent=`Latest historical date: ${latestHistoricalDate} • Historical observations used: ${historyRows.length}`;
  renderTable("historyTable",[
    {key:"Date",label:"Date"},{key:"Product",label:"Product"},{key:"Category",label:"Category"},
    {key:"Store",label:"Store"},{key:"Units_Sold",label:"Units Sold"},{key:"Price",label:"Price"},
    {key:"Discount_Percentage",label:"Discount %"} ,{key:"Promotion_Flag",label:"Promotion"},
    {key:"Stock_Availability",label:"Stock"},{key:"Day_of_Week",label:"Day"},
    {key:"Month",label:"Month"},{key:"Quarter",label:"Quarter"},{key:"Holiday_Flag",label:"Holiday"},
    {key:"Is_Weekend",label:"Weekend"},{key:"Season",label:"Season"},{key:"Weather",label:"Weather"},
    {key:"Local_Event_Flag",label:"Local Event"},{key:"Competitor_Price",label:"Competitor Price"},
    {key:"Economic_Indicator",label:"Economic Indicator"},{key:"Sales_Channel",label:"Channel"},
    {key:"Customer_Segment",label:"Customer Segment"},{key:"Marketing_Spend",label:"Marketing Spend"}
  ],histDisplay);
}

function drawActualForecast(canvasId,labels,actual,forecast){
  const canvas=$(canvasId), rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,pad={l:48,r:18,t:25,b:38};
  const all=actual.concat(forecast.filter(v=>v!==null));const max=Math.max(...all,1),min=Math.min(...all,0);
  const x=i=>pad.l+(i/Math.max(labels.length-1,1))*(w-pad.l-pad.r);
  const y=v=>h-pad.b-((v-min)/(max-min||1))*(h-pad.t-pad.b);
  ctx.strokeStyle="#e6e9ef";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const yy=pad.t+i*(h-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillStyle="#7a8190";ctx.font="11px Segoe UI";ctx.fillText(Math.round(max-(max-min)*i/4).toLocaleString(),5,yy+4)}
  function line(vals,stroke){
    ctx.strokeStyle=stroke;ctx.lineWidth=2.2;ctx.beginPath();let started=false;
    vals.forEach((v,i)=>{if(v===null)return;const px=x(i),py=y(v);if(!started){ctx.moveTo(px,py);started=true}else ctx.lineTo(px,py)});
    ctx.stroke();
  }
  line(actual,"#315efb");line(forecast,"#0f8a4b");
  const step=Math.max(1,Math.ceil(labels.length/7));ctx.fillStyle="#596273";ctx.font="11px Segoe UI";
  labels.forEach((lab,i)=>{if(i%step===0)ctx.fillText(lab.slice(5),x(i)-16,h-13)});
  ctx.fillStyle="#315efb";ctx.fillRect(pad.l,8,10,10);ctx.fillStyle="#596273";ctx.fillText("Actual Sales",pad.l+15,17);
  ctx.fillStyle="#0f8a4b";ctx.fillRect(pad.l+95,8,10,10);ctx.fillStyle="#596273";ctx.fillText("Forecast",pad.l+110,17);
}

async function loadAssets(){
  const [dataRes,featRes,modelRes]=await Promise.all([
    fetch("data/dataset.json"),
    fetch("model/feature_columns.json"),
    fetch("model/model.json")
  ]);
  if(!dataRes.ok||!featRes.ok||!modelRes.ok) throw new Error("Could not load one or more deployment files.");
  DATA=await dataRes.json();
  FEATURES=await featRes.json();
  MODEL=await modelRes.json();
  $("featureCount").textContent=FEATURES.length;
  $("datasetRows").textContent=DATA.length.toLocaleString("en-IN");
  $("loadStatus").textContent="✓ Model ready";
  $("loadStatus").classList.add("ready");
  initializeUI();
}

async function runForecast(){
  const btn=$("predictBtn");
  btn.disabled=true;btn.textContent="⏳ Generating forecast…";
  try{
    const start=$("startDate").value;
    const horizon=Number($("horizon").value);
    const latest=DATA.map(r=>r.Date).sort().at(-1);
    if(!start) throw new Error("Please select a forecast start date.");
    if(start<=latest) throw new Error(`Forecast start date must be after the latest historical date (${latest}).`);
    const userInputs=userInputObject();

    if(forecastType==="product"){
      const product=$("product").value, store=$("store").value;
      const forecast=generateForecast(product,store,horizon,start,DATA,userInputs);
      const hist=pairRows(product,store);
      renderResults("product",forecast,hist.at(-1).Date,hist,[]);
    }else{
      const store=$("storeTotal").value;
      const result=generateStoreTotal(store,horizon,start);
      const storeHist=storeRows(store);
      renderResults("store",result.totals,storeHist.at(-1).Date,storeHist,result.skipped);
    }
  }catch(e){
    $("results").classList.add("hidden");
    $("welcome").classList.remove("hidden");
    alert("Forecast error: "+e.message);
  }finally{
    btn.disabled=false;btn.textContent="🔮 Predict Sales";
  }
}

loadAssets().catch(err=>{
  $("loadStatus").textContent="Model load failed";
  $("loadStatus").style.background="#fff0f0";
  $("loadStatus").style.color="#9b1c1c";
  alert("Deployment files could not be loaded. If you opened index.html directly, use a local web server or GitHub Pages.");
});

"use strict";

// The canvas projects backend cells. It never invents a tensor or a decoded field.
window.CubeLab = (() => {
  const lab = {
    dimensions:4, size:3, seed:7, axes:[0,1,2], fixed:[0,0,0,0],
    plane:[0,1], layerAxis:2, layer:0, display:"cells", projection:"ghosts",
    material:"glass", coverage:1, spread:1.15, zoom:1, yaw:-.62, pitch:.45, source:"smooth", sources:[],
    selected:0, page:0, result:null, undo:[], busy:false, recorded:null, recordedError:null, recordedAttempted:false,
    status:"Creating a four-dimensional cube with the local Python implementation.", error:false,
  };
  let observer = null, canvas = null, hitCells = [], drag = null, mounted = false;
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? "").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const number = value => typeof value === "number" && Number.isFinite(value) ? Math.abs(value)>0 && Math.abs(value)<.0001 ? value.toExponential(3) : Number(value.toPrecision(6)).toLocaleString("en-US", {maximumFractionDigits:6}) : value == null ? "Not recorded" : typeof value === "object" ? JSON.stringify(value) : String(value);
  const label = text => String(text).replace(/_/g," ").replace(/^./,c=>c.toUpperCase());
  const axesNames = () => lab.result?.state?.axis_names ?? Array.from({length:lab.dimensions},(_,i)=>`D${i+1}`);
  const axisName = index => axesNames()[index] ?? `D${index+1}`;
  const axisOptions = selected => axesNames().map((name,i)=>`<option value="${i}"${i===selected?" selected":""}>${escape(name)}</option>`).join("");
  const options = (items,selected) => items.map(([value,text])=>`<option value="${escape(value)}"${String(value)===String(selected)?" selected":""}>${escape(text)}</option>`).join("");
  const rgb = cell => (cell?.rgb ?? [115,130,155]).slice(0,3).map(v=>Math.max(0,Math.min(255,Math.round(v))));
  const cssRGB = cell => `rgb(${rgb(cell).join(" ")})`;
  const fullCoord = cell => cell.full_coordinate ?? cell.coordinate ?? [];
  const cellId = cell => cell.id ?? fullCoord(cell).reduce((index,value)=>index*lab.size+value,0);
  const isSlice = cell => fullCoord(cell).every((value,axis)=>lab.axes.includes(axis)||value===lab.fixed[axis]);
  function allCells() {
    return lab.display === "holographic" ? lab.result?.holographic?.cells ?? [] : lab.display === "reconstruction" ? lab.result?.holographic?.decoded_cells ?? [] : lab.result?.all_cells ?? lab.result?.view?.cells ?? [];
  }
  function visibleCells() { return allCells().filter(cell=>lab.projection==="ghosts"||isSlice(cell)); }
  function card(title,subtitle,body,extra="") {
    return `<section class="card ${extra}"><div class="card-header"><div><h2>${title}</h2><p>${subtitle}</p></div></div><div class="card-body">${body}</div></section>`;
  }
  function facts(entries) { return `<dl class="facts">${entries.map(([key,value])=>`<div><dt>${escape(key)}</dt><dd class="number">${escape(number(value))}</dd></div>`).join("")}</dl>`; }
  function recordedMarkup() {
    if(!lab.recorded) return `<p class="hint">${escape(lab.recordedError??"Loading the saved cube experiment artifact.")}</p>`;
    const run=lab.recorded, rows=run.experiments??[];
    if(!rows.length)return '<p class="hint">No trained cube reconstruction models in the saved artifact.</p>';
    const names={direct_rgb24:"Direct RGB24",holographic24:"Chromatic phase / magnitude24",numeric_polar24:"Numeric phase / magnitude24"};
    const table=`<div class="table-wrapper"><table><caption>${rows.length} trained models, ${(run.geometry??[]).length} geometry checks, ${(run.reconstruction??[]).length} field-erasure measurements</caption><thead><tr>${["Input field","Seed","Field bits","Retained field","Initial validation MSE","Final validation MSE","Train seconds","Train / validation samples","Checkpoint"].map(text=>`<th scope="col">${text}</th>`).join("")}</tr></thead><tbody>${rows.map(row=>`<tr><th scope="row">${escape(names[row.representation]??label(row.representation))}</th><td class="number">${row.seed}</td><td class="number">${number(row.metrics?.bits_per_field_value)}</td><td class="number">${number(row.metrics?.retained_fraction*100)}%</td><td class="number">${number(row.metrics?.initial_validation_loss)}</td><td class="number">${number(row.metrics?.validation_loss)}</td><td class="number">${number(row.metrics?.training_seconds)}</td><td class="number">${number(row.dataset?.train_samples)} / ${number(row.dataset?.validation_samples)}</td><td>${row.artifact_path?`<a href="/api/artifact?path=${encodeURIComponent(row.artifact_path)}" download>Download</a>`:"Not recorded"}</td></tr>`).join("")}</tbody></table></div>`;
    const curves=rows.filter(row=>Array.isArray(row.curves)&&row.curves.length).map(row=>({...row,points:row.curves.filter(point=>Number.isFinite(point.val_loss))}));
    let plot="";
    if(curves.length){
      const width=620,height=205,pad=36,maxEpoch=Math.max(...curves.flatMap(row=>row.points.map(point=>point.epoch))),maxLoss=Math.max(...curves.flatMap(row=>row.points.map(point=>point.val_loss))),colors=["#4b9b87","#9666e3","#cd824f","#3090bd","#c95f99","#8c9247"];
      plot=`<details class="section-gap"><summary class="hint">View recorded validation curves</summary><svg class="chart" viewBox="0 0 ${width} ${height}" role="img"><title>Recorded cube reconstruction validation MSE over epochs</title>${Array.from({length:4},(_,i)=>{const y=height-pad-i*(height-pad*2)/3;return `<line class="gridline" x1="${pad}" y1="${y}" x2="${width-pad}" y2="${y}"/><text x="${pad-5}" y="${y+3}" text-anchor="end">${number(maxLoss*i/3)}</text>`;}).join("")}${curves.map((row,index)=>`<path d="${row.points.map((point,i)=>`${i?"L":"M"}${pad+point.epoch/maxEpoch*(width-pad*2)},${height-pad-point.val_loss/maxLoss*(height-pad*2)}`).join(" ")}" fill="none" stroke="${colors[index%colors.length]}" stroke-width="2"${row.seed!==7?' stroke-dasharray="4 3"':""}/>`).join("")}<text x="${pad}" y="${height-10}">Epoch 0</text><text x="${width-pad}" y="${height-10}" text-anchor="end">Epoch ${maxEpoch}</text></svg><div class="legend">${curves.map((row,index)=>`<span><i style="background:${colors[index%colors.length]}"></i>${escape(names[row.representation]??label(row.representation))} · seed ${row.seed}</span>`).join("")}</div></details>`;
    }
    return table+plot+`<p class="hint section-gap">${escape(rows[0].dataset?.source??"Dataset source not recorded")}. ${escape(rows[0].dataset?.training_scope??"")}. ${escape(rows[0].dataset?.generalization_scope??"")}. The scalar and spectral inputs have equal 24-bit field budgets. Spectral amplitude-scale metadata and the erasure mask cost extra storage.</p>`+(run.conclusions??[]).map(text=>`<p class="hint">${escape(text)}</p>`).join("");
  }
  function statusText(text,error=false) {
    lab.status=text; lab.error=error;
    if($("cube-status")) { $("cube-status").textContent=text; $("cube-status").classList.toggle("error",error); }
    if($("announcer")) $("announcer").textContent=text;
  }
  function cubeMarkup() {
    const shape = lab.result?.state?.shape ?? Array(lab.dimensions).fill(lab.size);
    const total = shape.reduce((a,b)=>a*b,1), cells = visibleCells(), hidden = lab.dimensions-3;
    const spectral = lab.display === "holographic";
    const disabled = lab.busy ? " disabled" : "";
    const hiddenFields = Array.from({length:lab.dimensions},(_,axis)=>axis).filter(axis=>!lab.axes.includes(axis)).map(axis=>`<div class="field"><label for="cube-fixed-${axis}">${escape(axisName(axis))} slice <output id="cube-fixed-label-${axis}" for="cube-fixed-${axis}">${lab.fixed[axis]}</output></label><input id="cube-fixed-${axis}" data-cube-fixed="${axis}" type="range" min="0" max="${lab.size-1}" step="1" value="${lab.fixed[axis]}"${disabled}></div>`).join("");
    const dimensionControls = `<form id="cube-create-form"><div class="toolbar"><div class="field"><label for="cube-dimensions">Dimensions</label><select id="cube-dimensions"${disabled}>${options([[3,"3 dimensions"],[4,"4 dimensions"],[5,"5 dimensions"],[6,"6 dimensions"]],lab.dimensions)}</select></div><div class="field"><label for="cube-size">Cells per axis</label><select id="cube-size"${disabled}>${options([[2,"2"],[3,"3"],[4,"4"]],lab.size)}</select></div></div><div class="field cube-control-gap"><label for="cube-source">Tensor source</label><select id="cube-source"${disabled}>${options([["smooth","Periodic demonstration"],["noise","Seeded random demonstration"],...lab.sources.map(sample=>[sample.name,`Recorded · ${sample.name} · ${sample.count} values`])],lab.source)}</select></div><div class="field cube-control-gap"><label for="cube-seed">Tensor seed</label><input id="cube-seed" type="number" min="0" max="2147483647" step="1" value="${lab.seed}" required${disabled}></div><div class="cube-control-actions"><button class="button primary" type="submit"${disabled}>Create cube</button><button class="button" type="button" id="cube-export"${!lab.result||lab.busy?" disabled":""}>Export state</button></div><p class="hint">Every cubie stores a real RGB code and a persistent identity. ${lab.size}<sup>${lab.dimensions}</sup> = ${total.toLocaleString()} cells in this cube.</p><p class="hint">Recorded samples must contain at least that many values. The cube uses their first values and preserves the sample name.</p></form>`;
    const viewControls = `<div class="toolbar">${["X","Y","Z"].map((name,i)=>`<div class="field"><label for="cube-axis-${i}">${name} axis</label><select id="cube-axis-${i}" data-cube-axis="${i}"${disabled}>${axisOptions(lab.axes[i])}</select></div>`).join("")}</div><div class="field cube-control-gap"><label for="cube-projection">Hidden dimensions</label><select id="cube-projection"${disabled}>${options([["ghosts","Project every slice"],["slice","Show selected slice"]],lab.projection)}</select></div><div class="cube-slice-fields">${hiddenFields||'<p class="hint">All three dimensions are visible.</p>'}</div><p class="hint">${hidden?`${hidden===1?"The extra axis is a real tensor dimension.":"The extra axes are real tensor dimensions."} Sliders select a slice. Projected copies retain all cells.`:"Change the dimension count to inspect hidden slices."}</p><div class="field cube-control-gap"><label for="cube-spread">Slice separation</label><input id="cube-spread" type="range" min="0" max="2.8" step=".05" value="${lab.spread}"></div>`;
    const fieldControls = `<div class="field"><label for="cube-display">Representation</label><select id="cube-display"${disabled}>${options([["cells","RGB cubies"],["holographic","Holographic spectral field"],["reconstruction","Decoded spectral reconstruction"]],lab.display)}</select></div><div class="field cube-control-gap"><label for="cube-material">Draw cells as</label><select id="cube-material">${options([["glass","Translucent cubes"],["solid","Solid cubes"],["points","Luminous points"]],lab.material)}</select></div><div class="field cube-control-gap"><label for="cube-coverage">Requested spectral coverage <output id="cube-coverage-label" for="cube-coverage">${Math.round(lab.coverage*100)}%</output></label><input id="cube-coverage" type="range" min=".05" max="1" step=".05" value="${lab.coverage}"${disabled}></div><div class="cube-control-actions"><button class="button" id="cube-decode"${!lab.result||lab.busy?" disabled":""}>Decode field</button></div><div class="cube-hint-block"><p>The spectral field is a unitary Fourier transform of the decoded RGB tensor. Phase sets hue, magnitude sets brightness.</p><p>Each coefficient contains contributions from all positions. Decode measures the actual reconstruction after coefficient erasure. The FFT uses floating point.</p></div>`;
    const twistControls = `<form id="cube-twist-form"><div class="toolbar"><div class="field"><label for="cube-plane-a">Plane axis A</label><select id="cube-plane-a"${disabled}>${axisOptions(lab.plane[0])}</select></div><div class="field"><label for="cube-plane-b">Plane axis B</label><select id="cube-plane-b"${disabled}>${axisOptions(lab.plane[1])}</select></div><div class="field"><label for="cube-layer-axis">Layer axis</label><select id="cube-layer-axis"${disabled}>${axisOptions(lab.layerAxis)}</select></div><div class="field"><label for="cube-layer">Layer index</label><select id="cube-layer"${disabled}>${options(Array.from({length:lab.size},(_,i)=>[i,i]),lab.layer)}</select></div></div><div class="cube-control-actions"><button class="button primary" type="submit" data-cube-turn="1"${!lab.result||lab.busy?" disabled":""}>Turn +90°</button><button class="button" type="button" id="cube-turn-back"${!lab.result||lab.busy?" disabled":""}>Turn -90°</button><button class="button" type="button" id="cube-undo"${!lab.undo.length||lab.busy?" disabled":""}>Undo turn</button><button class="button" type="button" id="cube-reset"${!lab.result||lab.busy?" disabled":""}>Reset cube</button></div><p class="hint">A turn permutes integer codes and cubie identities in the selected plane and layer. It preserves every stored code. Choose three different axes.</p></form>`;
    const axesText = lab.axes.map(axisName).join(" / ");
    const reconstructed = lab.display === "reconstruction";
    const sceneName = spectral?"Distributed spectral field":reconstructed?"Decoded field reconstruction":"Multidimensional chromatic cube";
    const fieldCoverage = lab.result?.holographic?.coverage ?? lab.coverage;
    return `<div class="cube-lab"><div class="cube-intro"><p><strong>A color cube with ${lab.dimensions} dimensions.</strong> ${hidden?`The ${hidden===1?"fourth dimension appears":"hidden dimensions appear"} as related cubes. Change the axes to see a different three-dimensional view of the same tensor.`:"Orbit the complete cube and turn a layer."}</p><span class="cube-live">${lab.result?"Live tensor state":"Local Python API"}</span></div><p class="hint" id="cube-provenance">Source ${escape(lab.result?.source??lab.result?.state?.source??"Waiting for backend provenance")}</p><div class="cube-stats"><div class="cube-stat"><span>Full tensor</span><strong>${escape(shape.join(" × "))}</strong><small>${total.toLocaleString()} distinct positions</small></div><div class="cube-stat"><span>Currently drawn</span><strong>${cells.length.toLocaleString()}</strong><small>${lab.projection==="slice"?"One selected 3D slice":"All tensor slices projected"}</small></div><div class="cube-stat"><span>Stored cell codes</span><strong>RGB24</strong><small>3 integer channels per position</small></div><div class="cube-stat"><span>${spectral||reconstructed?"Retained field":"Hidden dimensions"}</span><strong>${spectral||reconstructed?`${number(fieldCoverage*100)}%`:hidden}</strong><small>${spectral||reconstructed?"Actual retained coefficients":"Change axes or select a slice"}</small></div></div><p class="cube-status ${lab.error?"error":""}" id="cube-status">${escape(lab.status)}</p><div class="cube-layout"><figure class="cube-scene"><div class="cube-scene-top"><strong>${sceneName}</strong><span>${lab.dimensions}D · ${escape(axesText)} · ${lab.projection==="slice"?"slice":"full projection"}</span></div><div class="cube-stage"><canvas id="cube-canvas" class="cube-canvas" tabindex="0" role="img" aria-label="Interactive ${lab.dimensions}-dimensional ${spectral?"spectral field":"color cube"}. Arrow keys orbit, plus and minus zoom, brackets select cells. A numerical table follows." aria-describedby="cube-canvas-help"></canvas><div class="cube-stage-label">${spectral?"Hue = phase · brightness = magnitude":reconstructed?"Reconstructed numerical values · floating point inverse FFT":"Integer RGB codes · persistent cubie identities"}</div></div><div class="cube-camera-bar"><button class="button small" id="cube-orbit-left" aria-label="Orbit camera left">Rotate left</button><button class="button small" id="cube-orbit-right" aria-label="Orbit camera right">Rotate right</button><button class="button small" id="cube-camera-reset">Reset camera</button><div class="field"><label for="cube-zoom">Camera zoom</label><input id="cube-zoom" type="range" min=".45" max="2" step=".05" value="${lab.zoom}"></div></div><figcaption id="cube-canvas-help">Drag to orbit. Click a cell to inspect it. Arrow keys orbit, + and - zoom, [ and ] select cells. ${lab.projection==="ghosts"?"The bright cube is the selected slice. Faint copies show the remaining slices.":"Only cells at the hidden-axis slider values appear."} This is a geometric projection with translucent rendering.</figcaption></figure><div class="cube-controls">${card("Cube dimensions","Create a tensor with up to 4,096 cells",dimensionControls)}${card("Axes and slices","Choose which dimensions occupy the screen",viewControls)}${card("Holographic view","Inspect a global transform of the tensor",fieldControls)}</div></div>${card("Turn a tensor layer","A Rubik-style move in any pair of dimensions",twistControls,"cube-transform-card")}<div class="cube-data-layout">${card("Selected cell","A numerical view of the visible cell",'<div id="cube-inspector"></div>')}${card("Measured state","Execution and reconstruction evidence",'<div class="cube-metric-list" id="cube-metrics"></div>')}</div>${card("Cell data","Actual coordinates, RGB codes, and numerical values",'<div id="cube-data"></div>')}</div>`;
  }
  function unmount() { observer?.disconnect(); observer=null; canvas=null; hitCells=[]; drag=null; mounted=false; }
  function markup() { return cubeMarkup()+card("Recorded cube learning","Saved models trained to reconstruct structured fields after erasure",recordedMarkup(),"cube-recorded-card")+'<dialog id="cube-export-dialog" class="cube-export-dialog" aria-labelledby="cube-export-heading"><h2 id="cube-export-heading">Cube state JSON</h2><p class="hint">The state contains the current integer codes, cell identities, codec configuration, provenance, and measured metrics.</p><label for="cube-export-json">Export contents</label><textarea id="cube-export-json" readonly rows="14"></textarea><div class="cube-control-actions"><a id="cube-export-download" class="button primary" download>Download JSON</a><form method="dialog"><button class="button" type="submit">Close export</button></form></div></dialog>'; }
  function rerender(focusId) {
    if(!mounted||!$("cube-canvas")) return;
    unmount(); $("view").innerHTML=markup(); mount();
    if(focusId) $(focusId)?.focus({preventScroll:true});
  }
  async function request(operation,extra={},focusId) {
    if(lab.busy) return;
    lab.busy=true; lab.error=false;
    const previous=lab.result?.state;
    lab.status=operation==="twist"?"Applying an integer layer permutation.":operation==="decode"?"Decoding the retained Fourier coefficients.":operation==="create"?"Creating the tensor with Python.":"Updating the tensor view.";
    rerender();
    try {
      const response=await fetch("/api/cube",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({operation,state:operation==="create"?undefined:previous,dimensions:lab.dimensions,size:lab.size,seed:lab.seed,signal:lab.source==="noise"?"noise":"smooth",source_key:["smooth","noise"].includes(lab.source)?undefined:lab.source,view_axes:lab.axes,fixed:lab.fixed,display_mode:lab.display==="reconstruction"?"holographic":lab.display,coverage:lab.coverage,...extra})});
      const result=await response.json();
      if(!response.ok) throw new Error(result.error??`Cube request failed with status ${response.status}`);
      if(!result.state?.shape||!Array.isArray(result.view?.cells)) throw new Error("The cube API returned an incomplete tensor state.");
      if(operation==="twist"&&previous) { lab.undo.push(previous); if(lab.undo.length>32)lab.undo.shift(); }
      if(operation==="create"||operation==="reset") lab.undo=[];
      lab.result=result; lab.dimensions=result.state.shape.length; lab.size=result.state.shape[0];
      lab.selected=0; lab.page=0;
      lab.status=operation==="twist"?`Turned the ${axisName(lab.plane[0])} / ${axisName(lab.plane[1])} plane in ${axisName(lab.layerAxis)} layer ${lab.layer}. Codes and identities moved together.`:operation==="decode"?`Decoded ${Math.round(lab.coverage*100)}% spectral coverage. Reconstruction measurements appear below.`:operation==="reset"?"Restored the initial tensor layout.":operation==="create"?`Created ${result.state.shape.reduce((a,b)=>a*b,1).toLocaleString()} cells across ${lab.dimensions} dimensions.`:"Updated the view of the same tensor state.";
    } catch(error) {
      lab.status=error.message;lab.error=true;
      if(lab.result){lab.dimensions=lab.result.state.shape.length;lab.size=lab.result.state.shape[0];lab.axes=lab.axes.some(axis=>axis>=lab.dimensions)?[0,1,2]:lab.axes;lab.fixed=Array.from({length:lab.dimensions},(_,axis)=>Math.min(lab.size-1,lab.fixed[axis]??0));}
    }
    finally { lab.busy=false; rerender(focusId); statusText(lab.status,lab.error); }
  }
  function mount() {
    canvas=$("cube-canvas"); if(!canvas) return;
    mounted=true;
    const on=(id,event,fn)=>$(id)?.addEventListener(event,fn);
    on("cube-create-form","submit",event=>{event.preventDefault();lab.dimensions=Number($("cube-dimensions").value);lab.size=Number($("cube-size").value);lab.seed=Number($("cube-seed").value);lab.source=$("cube-source").value;lab.fixed=Array(lab.dimensions).fill(0);lab.axes=[0,1,2];lab.plane=[0,1];lab.layerAxis=2;lab.layer=0;request("create",{},"cube-dimensions");});
    document.querySelectorAll("[data-cube-axis]").forEach(select=>select.addEventListener("change",event=>{
      const slot=Number(event.target.dataset.cubeAxis),axis=Number(event.target.value),other=lab.axes.indexOf(axis),old=lab.axes[slot];
      lab.axes[slot]=axis; if(other!==-1&&other!==slot)lab.axes[other]=old;
      request("view",{},event.target.id);
    }));
    document.querySelectorAll("[data-cube-fixed]").forEach(input=>input.addEventListener("input",event=>{
      const axis=Number(event.target.dataset.cubeFixed);lab.fixed[axis]=Number(event.target.value);$("cube-fixed-label-"+axis).textContent=lab.fixed[axis];draw();updateInspector();
    }));
    document.querySelectorAll("[data-cube-fixed]").forEach(input=>input.addEventListener("change",event=>request("view",{},event.target.id)));
    on("cube-projection","change",event=>{lab.projection=event.target.value;lab.selected=0;lab.page=0;rerender(event.target.id);});
    on("cube-spread","input",event=>{lab.spread=Number(event.target.value);draw();});
    on("cube-material","change",event=>{lab.material=event.target.value;draw();});
    on("cube-display","change",event=>{lab.display=event.target.value;request("view",{},event.target.id);});
    on("cube-coverage","input",event=>{lab.coverage=Number(event.target.value);$("cube-coverage-label").textContent=`${Math.round(lab.coverage*100)}%`;});
    on("cube-coverage","change",event=>request("decode",{},event.target.id));
    on("cube-decode","click",()=>request("decode",{},"cube-decode"));
    const readTurn=()=>{lab.plane=[Number($("cube-plane-a").value),Number($("cube-plane-b").value)];lab.layerAxis=Number($("cube-layer-axis").value);lab.layer=Number($("cube-layer").value);};
    const turn=turns=>{readTurn();if(new Set([...lab.plane,lab.layerAxis]).size!==3){statusText("Choose different axes for the plane and layer.",true);return;}request("twist",{axes:lab.plane,layer_axis:lab.layerAxis,layer_index:lab.layer,turns},turns===1?"cube-plane-a":"cube-turn-back");};
    ["cube-plane-a","cube-plane-b","cube-layer-axis","cube-layer"].forEach(id=>on(id,"change",readTurn));
    on("cube-twist-form","submit",event=>{event.preventDefault();turn(1);});on("cube-turn-back","click",()=>turn(-1));
    on("cube-reset","click",()=>request("reset",{},"cube-reset"));
    on("cube-undo","click",async()=>{const previous=lab.undo.at(-1);if(!previous)return;await request("view",{state:previous},"cube-undo");if(!lab.error){lab.undo.pop();statusText("Restored the state before the last turn.");rerender("cube-plane-a");}});
    on("cube-export","click",()=>{if(!lab.result)return;const data={schema_version:1,state:lab.result.state,view:{axes:lab.axes,fixed:lab.fixed,projection:lab.projection,display:lab.display,coverage:lab.coverage},metrics:lab.result.metrics,holographic_metrics:lab.result.holographic?.metrics};const json=JSON.stringify(data,null,2),url=URL.createObjectURL(new Blob([json],{type:"application/json"})),dialog=$("cube-export-dialog"),a=$("cube-export-download");$("cube-export-json").value=json;a.href=url;a.download=`chromatensor-cube-${lab.dimensions}d-${lab.size}.json`;dialog.addEventListener("close",()=>URL.revokeObjectURL(url),{once:true});dialog.showModal();statusText("Prepared the current cube state as JSON. Download it or copy the export contents.");});
    on("cube-orbit-left","click",()=>{lab.yaw-=.22;draw();});on("cube-orbit-right","click",()=>{lab.yaw+=.22;draw();});
    on("cube-camera-reset","click",()=>{lab.yaw=-.62;lab.pitch=.45;lab.zoom=1;$("cube-zoom").value=1;draw();});
    on("cube-zoom","input",event=>{lab.zoom=Number(event.target.value);draw();});
    canvas.addEventListener("pointerdown",event=>{if(event.button!==0)return;drag={x:event.clientX,y:event.clientY,moved:false};canvas.setPointerCapture(event.pointerId);});
    canvas.addEventListener("pointermove",event=>{if(!drag)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>1)drag.moved=true;lab.yaw+=dx*.009;lab.pitch=Math.max(-1.3,Math.min(1.3,lab.pitch+dy*.009));drag.x=event.clientX;drag.y=event.clientY;draw();});
    canvas.addEventListener("pointerup",event=>{if(drag&&!drag.moved){const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;const hits=hitCells.filter(cell=>Math.hypot(cell.x-x,cell.y-y)<cell.radius).sort((a,b)=>b.depth-a.depth);if(hits.length){lab.selected=hits[0].index;updateInspector();draw();}}drag=null;});
    canvas.addEventListener("pointercancel",()=>{drag=null;});
    canvas.addEventListener("keydown",event=>{
      const cells=visibleCells(); let changed=true;
      if(event.key==="ArrowLeft")lab.yaw-=.12;else if(event.key==="ArrowRight")lab.yaw+=.12;else if(event.key==="ArrowUp")lab.pitch=Math.max(-1.3,lab.pitch-.12);else if(event.key==="ArrowDown")lab.pitch=Math.min(1.3,lab.pitch+.12);else if(["+","="].includes(event.key))lab.zoom=Math.min(2,lab.zoom+.1);else if(event.key==="-")lab.zoom=Math.max(.45,lab.zoom-.1);else if(event.key==="]")lab.selected=Math.min(cells.length-1,lab.selected+1);else if(event.key==="[")lab.selected=Math.max(0,lab.selected-1);else changed=false;
      if(changed){event.preventDefault();$("cube-zoom").value=lab.zoom;draw();updateInspector(true);}
    });
    observer=new ResizeObserver(()=>draw());observer.observe(canvas);
    updateInspector();updateMetrics();updateTable();draw();
    if(!lab.result&&!lab.busy)request("create");
    if(!lab.recordedAttempted)loadRecorded();
  }
  async function loadRecorded() {
    lab.recordedAttempted=true;
    try {const response=await fetch("/api/cube-results",{cache:"no-store"});const data=await response.json();if(!response.ok)throw new Error(data.error??"The saved cube experiment could not be loaded.");lab.recorded=data;}
    catch(error){lab.recordedError=error.message;}
    const target=document.querySelector(".cube-recorded-card .card-body");if(target)target.innerHTML=recordedMarkup();
  }
  function updateInspector(announce=false) {
    const cells=visibleCells();lab.selected=Math.max(0,Math.min(cells.length-1,lab.selected));const cell=cells[lab.selected];
    if(!$("cube-inspector"))return;
    if(!cell){$("cube-inspector").innerHTML='<p class="hint">No backend cells available for this view.</p>';return;}
    const spectral=lab.display==="holographic",reconstructed=lab.display==="reconstruction",coord=fullCoord(cell);
    $("cube-inspector").innerHTML=`<div class="cube-inspector-top"><div class="cube-cell-color" style="background:${cssRGB(cell)}"></div><p>${spectral?"Field coefficient":reconstructed?"Reconstructed cell":"Cubie"} ${escape(cellId(cell))}<small>${coord.map((value,axis)=>`${escape(axisName(axis))} ${value}`).join(" · ")}</small></p></div><div class="field"><label for="cube-cell-select">Visible cell</label><select id="cube-cell-select">${cells.map((item,index)=>`<option value="${index}"${index===lab.selected?" selected":""}>${escape(cellId(item))} · [${fullCoord(item).join(", ")}]</option>`).join("")}</select></div><div class="section-gap">${facts(spectral?[["Coordinate",coord],["Phase, radians",cell.phase],["Magnitude",cell.magnitude??cell.amplitude],["Coefficient retained",cell.kept??"Not recorded"],["Display RGB",rgb(cell)]]:[["Identity",cell.id],["Full coordinate",coord],["Decoded value",cell.value],[reconstructed?"Reconstruction display RGB":"Integer RGB code",cell.codes??cell.code??cell.rgb],["Selected hidden slice",isSlice(cell)?"Yes":"No"]])}</div>`;
    $("cube-cell-select").addEventListener("change",event=>{lab.selected=Number(event.target.value);draw();updateInspector(true);});
    if(announce&&$("announcer"))$("announcer").textContent=`${spectral?"Coefficient":"Cubie"} ${cellId(cell)}. Coordinate ${coord.join(", ")}. ${spectral?`Phase ${number(cell.phase)}, magnitude ${number(cell.magnitude??cell.amplitude)}.`:`Decoded value ${number(cell.value)}.`}`;
  }
  function updateMetrics() {
    const target=$("cube-metrics");if(!target)return;
    if(!lab.result){target.innerHTML='<p class="hint">Measurements appear after the local API returns a tensor.</p>';return;}
    const base=Object.entries(lab.result.metrics??{}).filter(([,value])=>typeof value!=="object"||value===null);
    const field=Object.entries(lab.result.holographic?.metrics??{}).filter(([,value])=>typeof value!=="object"||value===null);
    target.innerHTML=facts(base.map(([key,value])=>[label(key),value]))+(field.length?`<h3 class="section-gap">Spectral reconstruction</h3>${facts(field.map(([key,value])=>[label(key),value]))}`:"")+`<p class="hint section-gap">The cube stores integer RGB codes. Layer turns only permute those codes. Fourier encoding and reconstruction use numerical floating point arithmetic.</p>`;
  }
  function updateTable() {
    const target=$("cube-data");if(!target)return;const cells=visibleCells(),limit=64,pages=Math.max(1,Math.ceil(cells.length/limit));lab.page=Math.min(lab.page,pages-1);
    const spectral=lab.display==="holographic",reconstructed=lab.display==="reconstruction",rows=cells.slice(lab.page*limit,(lab.page+1)*limit);
    target.innerHTML=`<div class="cube-table-controls"><span>${cells.length?lab.page*limit+1:0} to ${Math.min(cells.length,(lab.page+1)*limit)} of ${cells.length.toLocaleString()} visible cells</span><div><button class="button small" id="cube-page-back"${lab.page===0?" disabled":""}>Previous cells</button> <button class="button small" id="cube-page-next"${lab.page===pages-1?" disabled":""}>Next cells</button></div></div><div class="table-wrapper"><table><caption>${spectral?"Measured spectral coefficients":reconstructed?"Inverse FFT values and reconstruction display colors":"Decoded tensor values and actual RGB codes"}</caption><thead><tr><th scope="col">${spectral?"Coefficient":"Cubie ID"}</th><th scope="col">Full coordinate</th><th scope="col">${spectral?"Phase, radians":"Decoded value"}</th><th scope="col">${spectral?"Magnitude":reconstructed?"Display RGB":"RGB code"}</th><th scope="col">${spectral?"Retained":"Selected slice"}</th></tr></thead><tbody>${rows.map((cell,i)=>`<tr><th scope="row"><button class="table-link" data-cube-cell="${lab.page*limit+i}">${escape(cellId(cell))}</button></th><td class="number">[${fullCoord(cell).join(", ")}]</td><td class="number">${escape(number(spectral?cell.phase:cell.value))}</td><td class="number">${escape(number(spectral?cell.magnitude??cell.amplitude:cell.codes??cell.code??cell.rgb))}</td><td>${spectral?cell.kept===false?"Erased":"Retained":isSlice(cell)?"Yes":"No"}</td></tr>`).join("")}</tbody></table></div>`;
    $("cube-page-back").addEventListener("click",()=>{lab.page--;updateTable();$("cube-page-back")?.focus();});$("cube-page-next").addEventListener("click",()=>{lab.page++;updateTable();$("cube-page-next")?.focus();});
    document.querySelectorAll("[data-cube-cell]").forEach(button=>button.addEventListener("click",()=>{lab.selected=Number(button.dataset.cubeCell);draw();updateInspector(true);}));
  }
  function draw() {
    if(!canvas)return;const bounds=canvas.getBoundingClientRect();if(!bounds.width||!bounds.height)return;
    const dpr=Math.min(window.devicePixelRatio||1,2),width=bounds.width,height=bounds.height;
    if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
    const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    const backdrop=ctx.createRadialGradient(width*.5,height*.45,10,width*.5,height*.5,Math.max(width,height)*.7);backdrop.addColorStop(0,"#182843");backdrop.addColorStop(1,"#07101e");ctx.fillStyle=backdrop;ctx.fillRect(0,0,width,height);
    const cells=visibleCells(),center=(lab.size-1)/2,hidden=Array.from({length:lab.dimensions},(_,i)=>i).filter(i=>!lab.axes.includes(i));
    const ca=Math.cos(lab.yaw),sa=Math.sin(lab.yaw),cp=Math.cos(lab.pitch),sp=Math.sin(lab.pitch);
    const transform=point=>{const x=point[0]*ca+point[2]*sa,z=-point[0]*sa+point[2]*ca;return[x,point[1]*cp-z*sp,point[1]*sp+z*cp];};
    const extent=lab.size+hidden.length*lab.spread*lab.size*.5,camera=Math.max(9,extent*3.5),scale=Math.min(width,height)*.55/extent*lab.zoom;
    const project=point=>{const [x,y,z]=transform(point),perspective=camera/(camera-z);return{x:width/2+x*scale*perspective,y:height*.53-y*scale*perspective,depth:z,scale:scale*perspective};};
    const pointFor=cell=>{const coord=fullCoord(cell),point=lab.axes.map(axis=>(coord[axis]??0)-center);if(lab.projection==="ghosts")hidden.forEach((axis,index)=>{const offset=((coord[axis]??0)-center)*lab.spread;const angle=.55+index*2.05;point[0]+=Math.cos(angle)*offset*.94;point[1]+=Math.sin(angle)*offset*.86;point[2]+=(index%2?-.68:.68)*offset;});return point;};
    hitCells=[];
    ctx.strokeStyle="#40517055";ctx.lineWidth=1;
    for(let n=-4;n<=4;n++){const a=project([n,-lab.size*.75,-4]),b=project([n,-lab.size*.75,4]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();const c=project([-4,-lab.size*.75,n]),d=project([4,-lab.size*.75,n]);ctx.beginPath();ctx.moveTo(c.x,c.y);ctx.lineTo(d.x,d.y);ctx.stroke();}
    if(!cells.length){ctx.fillStyle="#d0dbee";ctx.textAlign="center";ctx.font='14px "Segoe UI", sans-serif';ctx.fillText(lab.error?"Cube API unavailable. See the message above.":"Waiting for the real tensor state.",width/2,height/2);return;}
    const faceVertices=[[[1,0,0],[[1,-1,-1],[1,1,-1],[1,1,1],[1,-1,1]]],[[-1,0,0],[[-1,-1,1],[-1,1,1],[-1,1,-1],[-1,-1,-1]]],[[0,1,0],[[-1,1,-1],[-1,1,1],[1,1,1],[1,1,-1]]],[[0,-1,0],[[-1,-1,1],[-1,-1,-1],[1,-1,-1],[1,-1,1]]],[[0,0,1],[[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]],[[0,0,-1],[[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]]];
    const faces=[],points=[],spectral=lab.display==="holographic",half=.35;
    cells.forEach((cell,index)=>{
      const p=pointFor(cell),screen=project(p),selected=index===lab.selected,focusSlice=isSlice(cell),kept=cell.kept!==false;
      hitCells.push({...screen,radius:Math.max(8,screen.scale*.53),index});
      const opacity=lab.projection==="ghosts"&&!focusSlice ? .23 : .66;
      if(lab.material==="points"||spectral){points.push({...screen,cell,index,selected,focusSlice,opacity:kept?opacity:.045});if(lab.material==="points")return;}
      faceVertices.forEach(([normal,verts])=>{
        const n=transform(normal);if(n[2]<=.02)return;
        const vertices=verts.map(v=>project(p.map((value,i)=>value+v[i]*half)));
        faces.push({vertices,depth:vertices.reduce((sum,v)=>sum+v.depth,0)/4,color:rgb(cell),selected,focusSlice,shade:.7+n[2]*.3,edgeOpacity:spectral&&!kept ? .05 : focusSlice ? .86 : .32,opacity:spectral ? kept ? .11 : .025 : lab.material==="solid" ? focusSlice||lab.projection==="slice" ? 1 : .48 : opacity*.57});
      });
    });
    faces.sort((a,b)=>a.depth-b.depth).forEach(face=>{
      ctx.beginPath();face.vertices.forEach((point,i)=>i?ctx.lineTo(point.x,point.y):ctx.moveTo(point.x,point.y));ctx.closePath();ctx.fillStyle=`rgba(${face.color.map(value=>Math.round(value*face.shade)).join(",")},${face.opacity})`;ctx.fill();ctx.lineWidth=face.selected?1.6:.7;ctx.strokeStyle=face.selected?"#f4edff":`rgba(${face.color.join(",")},${face.edgeOpacity})`;ctx.stroke();
    });
    points.sort((a,b)=>a.depth-b.depth).forEach(point=>{
      const radius=Math.max(1.7,point.scale*(spectral ? .12 : .17));ctx.globalAlpha=point.opacity;ctx.fillStyle=cssRGB(point.cell);ctx.shadowColor=cssRGB(point.cell);ctx.shadowBlur=point.focusSlice?12:4;ctx.beginPath();ctx.arc(point.x,point.y,radius,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0;ctx.globalAlpha=1;if(point.selected){ctx.strokeStyle="#f5ecff";ctx.lineWidth=1.7;ctx.beginPath();ctx.arc(point.x,point.y,radius+4,0,Math.PI*2);ctx.stroke();}
    });
    const origin=project([-lab.size*.78,-lab.size*.78,-lab.size*.78]);
    const axisColors=["#f4a4b8","#85dec1","#a2bbff"];
    lab.axes.forEach((axis,index)=>{const end=[-lab.size*.78,-lab.size*.78,-lab.size*.78];end[index]+=lab.size*.7;const target=project(end);ctx.strokeStyle=axisColors[index];ctx.lineWidth=1.8;ctx.beginPath();ctx.moveTo(origin.x,origin.y);ctx.lineTo(target.x,target.y);ctx.stroke();ctx.fillStyle=axisColors[index];ctx.font='11px "Segoe UI", sans-serif';ctx.textAlign="center";ctx.fillText(axisName(axis),target.x,target.y-8);});
    ctx.shadowBlur=0;ctx.globalAlpha=1;
  }
  function setResults(data) { lab.sources=(data?.tensor_samples??[]).filter(sample=>Array.isArray(sample.values)).map(sample=>({name:sample.name,count:sample.values.length})); }
  return {markup,mount,unmount,setResults};
})();

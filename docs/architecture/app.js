"use strict";
(() => {
  const graph = window.CHROMATIC_ARCHITECTURE;
  const byId = new Map(graph.nodes.map(node => [node.id, node]));
  const byGroup = new Map(graph.groups.map(group => [group.id, group]));
  const colors = {native:"#08766a",entry:"#375fa2",legacy:"#80583f",dashboard:"#746099",research:"#617c25",release:"#3b6778"};
  const $ = id => document.getElementById(id);
  const esc = value => String(value).replace(/[&<>"']/g, character => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[character]));
  const state = {view:"overview",subsystem:"all",flow:"all",edgeType:"all",search:"",layers:new Set(graph.groups.map(group=>group.id)),external:false,selected:null,zoom:1,panX:0,panY:0};
  let base = {width:1200,height:650}, visible = {nodes:[],edges:[]}, pan = null;
  const views = {
    overview:["One library, two mathematical paths","The native engine owns literal RGBD hues and computes integer phase algebra. Legacy PyTorch APIs decode numerical values. Select a subsystem to inspect its actual components."],
    data:["Where data goes and who owns it","Follow data, output ownership and file I/O. An arrow points toward the recipient. Native and legacy files carry different meanings for D and cannot be interchanged."],
    journeys:["Follow a complete call path","Choose a flow to see its components and reviewed relationships. The numbered cards show the journey order. Detail panels state inputs, outputs and side effects."],
    boundaries:["What must be trusted","Dashed relationships cross process, pointer, file, browser or dataset boundaries. Raw C pointer validity remains the caller's responsibility. The local dashboard has no accounts or authentication."],
    failures:["How a path rejects input","Each component lists concrete rejection conditions. The graph shows affected relationships; the text map below lists failure behavior for every visible component."]
  };
  function sourceLinks(references) {
    return `<ul class="source-links">${references.map(reference => `<li><a href="${esc(reference.url)}" target="_blank" rel="noopener">${esc(reference.path)}:${reference.line}${reference.symbol ? ` · ${esc(reference.symbol)}` : ""}</a><small>SHA-256 ${reference.sha256.slice(0,16)}</small></li>`).join("")}</ul>`;
  }
  $("source-identity").textContent = `${graph.nodes.length} components · ${graph.edges.length} reviewed relationships\nSource ${graph.source_identity.slice(0,16)}`;
  $("evidence-note").textContent = graph.evidence_note;
  $("status-note").textContent = graph.status_note;
  $("inventory-summary").textContent = `${graph.inventory.length} inventoried executable, build, test and lock files. ${graph.inventory.reduce((sum,file)=>sum+file.imports.length,0)} lexical imports are recorded separately in graph.json.`;
  $("subsystem").insertAdjacentHTML("beforeend", graph.groups.map(group=>`<option value="${group.id}">${esc(group.name)}</option>`).join(""));
  $("flow").insertAdjacentHTML("beforeend", graph.flows.map(flow=>`<option value="${flow.id}">${esc(flow.name)}</option>`).join(""));
  $("layers").innerHTML = graph.groups.map(group=>`<label class="check"><input type="checkbox" value="${group.id}" checked><span class="layer-dot dot-${group.id}"></span>${esc(group.name)}</label>`).join("");
  $("integration-list").innerHTML = graph.integrations.map(item=>`<article class="integration-item"><span class="integration-status">${item.status === "absent" ? "Absent in reviewed code" : "Present dependency"}</span><h3>${esc(item.name)}</h3><p>${esc(item.detail)}</p>${sourceLinks(item.sources)}</article>`).join("");
  // Color values are fixed local constants; no source or user text is used as markup or CSS.
  function matching() {
    const flow = graph.flows.find(flow=>flow.id===state.flow);
    const touchpoints = new Set(["datasets","build","python","ffi","http","browser","storage","legacy-storage"]);
    let nodes = graph.nodes.filter(node=>state.layers.has(node.group) && (state.subsystem==="all" || node.group===state.subsystem)
      && (!flow || flow.steps.includes(node.id)) && (!state.external || touchpoints.has(node.id))
      && (!state.search || JSON.stringify(node).toLowerCase().includes(state.search)));
    const ids = new Set(nodes.map(node=>node.id));
    let edges = graph.edges.filter(edge=>ids.has(edge.source)&&ids.has(edge.target)
      && (state.edgeType==="all" || edge.type===state.edgeType) && (!flow || edge.flows.includes(flow.id)));
    if(state.view==="boundaries") edges=edges.filter(edge=>edge.boundary);
    if(state.view==="data") edges=edges.filter(edge=>["data","read","write","ownership"].includes(edge.type));
    if(["boundaries","data"].includes(state.view) && state.subsystem==="all" && !state.search) {
      const involved=new Set(edges.flatMap(edge=>[edge.source,edge.target]));nodes=nodes.filter(node=>involved.has(node.id));
    }
    return {nodes,edges};
  }
  function resetCamera() { state.zoom=1;state.panX=0;state.panY=0;updateCamera(); }
  function updateCamera() {
    const width=base.width/state.zoom,height=base.height/state.zoom;
    $("map").setAttribute("viewBox",`${state.panX+(base.width-width)/2} ${state.panY+(base.height-height)/2} ${width} ${height}`);
    $("zoom-level").textContent=`${Math.round(state.zoom*100)}%`;
  }
  function zoomBy(factor) {state.zoom=Math.min(4,Math.max(.35,state.zoom*factor));updateCamera();}
  function graphMarkup(nodes, edges, collapsed) {
    let cards, relationships;
    if(collapsed) {
      cards=graph.groups.filter(group=>nodes.some(node=>node.group===group.id)).map(group=>({...group,group:group.id,isGroup:true}));
      const aggregate=new Map();
      edges.forEach(edge=>{const source=byId.get(edge.source).group,target=byId.get(edge.target).group;if(source===target)return;const key=`${source}:${target}`;const current=aggregate.get(key)||{source,target,labels:new Set(),boundary:false};current.labels.add(edge.type);current.boundary ||= edge.boundary;aggregate.set(key,current);});
      relationships=[...aggregate.values()].map(edge=>({...edge,label:[...edge.labels].join(" / ")}));
    } else {cards=nodes;relationships=edges;}
    const cols = Math.min(2, Math.max(1,cards.length));
    const width=collapsed?310:340,height=collapsed?140:93,gapX=40,gapY=collapsed?40:85;
    const rows=Math.ceil(cards.length/cols);
    base={width:Math.max(600,cols*(width+gapX)+70),height:Math.max(440,rows*(height+gapY)+80)};
    const positions=new Map(cards.map((card,index)=>[card.id,{x:55+(index%cols)*(width+gapX),y:60+Math.floor(index/cols)*(height+gapY),width,height}]));
    const paths=relationships.map((edge,index)=>{
      const a=positions.get(edge.source),b=positions.get(edge.target);if(!a||!b)return "";
      const ax=a.x+a.width/2,ay=a.y+a.height/2,bx=b.x+b.width/2,by=b.y+b.height/2;
      const horizontal=Math.abs(bx-ax)>Math.abs(by-ay);
      const sx=horizontal?ax+Math.sign(bx-ax)*a.width/2:ax,sy=horizontal?ay:ay+Math.sign(by-ay)*a.height/2;
      const tx=horizontal?bx-Math.sign(bx-ax)*b.width/2:bx,ty=horizontal?by:by-Math.sign(by-ay)*b.height/2;
      const shift=((index%3)-1)*10;
      const path=horizontal?`M ${sx} ${sy} C ${(sx+tx)/2} ${sy+shift}, ${(sx+tx)/2} ${ty+shift}, ${tx} ${ty}`:`M ${sx} ${sy} C ${sx+shift} ${(sy+ty)/2}, ${tx+shift} ${(sy+ty)/2}, ${tx} ${ty}`;
      const label=collapsed?edge.label:edge.type;
      return `<g><path class="graph-edge ${edge.boundary?"boundary":""}" d="${path}" marker-end="url(#arrow${edge.boundary?"-boundary":""})"><title>${esc(edge.label)}${edge.boundary?". Crosses a trust boundary.":""}</title></path><text class="edge-label" x="${(sx+tx)/2+shift}" y="${(sy+ty)/2-5}" text-anchor="middle">${esc(label)}</text></g>`;
    }).join("");
    const markup=cards.map((card,index)=>{
      const p=positions.get(card.id),selected=state.selected===card.id;
      const members=card.isGroup?nodes.filter(node=>node.group===card.id):[];
      const group=byGroup.get(card.group),flow=graph.flows.find(flow=>flow.id===state.flow);
      const order=flow?`${flow.steps.indexOf(card.id)+1}. `:"";
      const status=card.isGroup?["legacy","research"].includes(card.id)?"experimental":"implemented":card.status;
      const summary=card.isGroup?card.description:card.kind==="store"?"Persistent local data":card.trust_boundary.slice(0,40)+(card.trust_boundary.length>40?"…":"");
      const words=summary.split(" "),lines=[""];for(const word of words){const last=lines.length-1;if(lines[last].length+word.length>(card.isGroup?36:40))lines.push(word);else lines[last]+=(lines[last]?" ":"")+word;}
      return `<g class="node ${card.isGroup?"subsystem-card":""}" data-node="${card.id}" data-group="${card.isGroup}" tabindex="0" role="button" aria-label="${esc(card.name)}. ${status}.${card.isGroup?" Drill into subsystem":" Open details"}" aria-pressed="${selected}" transform="translate(${p.x} ${p.y})"><rect width="${p.width}" height="${p.height}" rx="7"/><path d="M 0 8 Q 0 0 8 0 L ${p.width-8} 0 Q ${p.width} 0 ${p.width} 8" fill="none" stroke="${colors[card.group]}" stroke-width="5"/><text class="node-kind" x="17" y="24">${esc(status.toUpperCase())}</text><text class="node-name" x="17" y="49">${esc(order+card.name)}</text>${lines.slice(0,card.isGroup?2:1).map((line,i)=>`<text class="node-summary" x="17" y="${73+i*19}">${esc(line)}</text>`).join("")}${card.isGroup?`<text class="node-count" x="17" y="123">${members.length} components · Select to drill down</text>`:""}</g>`;
    }).join("");
    return `<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#648075"/></marker><marker id="arrow-boundary" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9d683e"/></marker></defs>${paths}${markup}`;
  }
  function showDetails() {
    const node=byId.get(state.selected),group=byGroup.get(state.selected);
    if(node) {
      const fields=[["Inputs",node.inputs],["Outputs",node.outputs],["Side effects",node.side_effects],["Ownership",node.ownership],["Direct call dependencies",node.dependencies.map(id=>byId.get(id).name).join(", ")||"No direct call edge in this reviewed map."],["Trust boundary",node.trust_boundary],["Failure behavior",node.failures]];
      $("details").innerHTML=`<p class="eyebrow">${esc(byGroup.get(node.group).name)}</p><h2 id="details-title">${esc(node.name)}</h2><span class="badge">${esc(node.status)}</span><p>${esc(node.purpose)}</p><dl>${fields.map(([label,value])=>`<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join("")}</dl><h3>Source symbols</h3>${sourceLinks(node.sources)}<h3>Relevant verification code</h3>${sourceLinks(node.tests)}<p class="hint">Test links identify coverage. They do not claim a passing execution in this view.</p>`;
    } else if(group) {
      $("details").innerHTML=`<p class="eyebrow">Subsystem</p><h2 id="details-title">${esc(group.name)}</h2><p>${esc(group.description)}</p><div class="group-children">${graph.nodes.filter(node=>node.group===group.id).map(node=>`<button type="button" data-select="${node.id}">${esc(node.name)}</button>`).join("")}</div><p class="hint">Choose a component to read its ownership, failures and source symbols.</p>`;
    } else {
      $("details").innerHTML=`<p class="eyebrow">Read the code path</p><h2 id="details-title">Select a component</h2><p>Start with a subsystem card, search for a behavior or choose a flow. Every component has source references and explicit failure conditions.</p><h3>The key distinction</h3><p>Native D refines a hue step. Legacy RGBD D stores an IEEE exponent. Their bytes represent different mathematics.</p><p>The map contains no native neural training or GPU dispatch. Open integrations to inspect external dependencies and absent services.</p>`;
    }
    $("details").querySelectorAll("[data-select]").forEach(button=>button.addEventListener("click",()=>selectNode(button.dataset.select)));
  }
  function selectNode(id) {
    state.selected=id;showDetails();
    document.querySelectorAll("[data-node],[data-select]").forEach(element=>element.setAttribute("aria-pressed",String((element.dataset.node||element.dataset.select)===id)));
    const node=byId.get(id);if(node)$("map-status").textContent=`Selected ${node.name}. ${visible.nodes.length} visible components.`;
  }
  function drillGroup(id) {state.subsystem=id;state.selected=id;$("subsystem").value=id;render();}
  function render() {
    visible=matching();
    if(byId.has(state.selected)&&!visible.nodes.some(node=>node.id===state.selected))state.selected=null;
    const collapsed=state.subsystem==="all"&&state.flow==="all"&&((!state.search&&!state.external)||visible.nodes.length>8);
    $("map").innerHTML=graphMarkup(visible.nodes,visible.edges,collapsed);
    $("empty-map").hidden=visible.nodes.length>0;
    $("map-status").textContent=`${visible.nodes.length} components · ${visible.edges.length} reviewed relationships${collapsed?" · grouped map":""}`;
    $("text-count").textContent=`${visible.nodes.length} components in this view`;
    $("component-list").innerHTML=visible.nodes.map(node=>`<button type="button" data-select="${node.id}" aria-pressed="${state.selected===node.id}"><span class="layer-dot dot-${node.group}"></span>${esc(node.name)}</button>`).join("")||"<p>No matching components.</p>";
    $("relationship-list").innerHTML=visible.edges.map(edge=>`<tr><td><button type="button" data-select="${edge.source}">${esc(byId.get(edge.source).name)}</button></td><td><button type="button" data-select="${edge.target}">${esc(byId.get(edge.target).name)}</button></td><td>${esc(edge.type)}<br>${esc(edge.label)}</td><td>${edge.boundary?"Crosses a trust boundary":"Within declared ownership"}</td><td>${edge.evidence.map(reference=>`<a href="${esc(reference.url)}" target="_blank" rel="noopener">${esc(reference.path)}:${reference.line}</a>`).join("<br>")}</td></tr>`).join("")||'<tr><td colspan="5">No relationships match the current filters. Components can still be inspected.</td></tr>';
    $("failure-list").innerHTML=state.view==="failures"?`<h3>Failure behavior by component</h3><div class="failure-grid">${visible.nodes.map(node=>`<article><h3>${esc(node.name)}</h3><p>${esc(node.failures)}</p></article>`).join("")}</div>`:"";
    const [title,description]=views[state.view],flow=graph.flows.find(flow=>flow.id===state.flow);
    $("view-title").textContent=flow?flow.name:title;$("view-description").textContent=flow?flow.description:description;
    document.querySelectorAll("[data-view]").forEach(button=>button.setAttribute("aria-pressed",String(button.dataset.view===state.view)));
    $("map").querySelectorAll("[data-node]").forEach(element=>{
      const activate=()=>element.dataset.group==="true"?drillGroup(element.dataset.node):selectNode(element.dataset.node);
      element.addEventListener("click",activate);element.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();event.stopPropagation();activate();}});
    });
    $("text-map").querySelectorAll("[data-select]").forEach(button=>button.addEventListener("click",()=>selectNode(button.dataset.select)));
    showDetails();resetCamera();
  }
  document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{state.view=button.dataset.view;if(state.view==="journeys"){if(state.flow==="all"){state.flow="native";$("flow").value="native";}state.subsystem="all";$("subsystem").value="all";}else{state.flow="all";$("flow").value="all";}state.selected=null;render();}));
  $("search").addEventListener("input",()=>{state.search=$("search").value.trim().toLowerCase();render();});
  $("subsystem").addEventListener("change",()=>{state.subsystem=$("subsystem").value;state.selected=state.subsystem==="all"?null:state.subsystem;render();});
  $("flow").addEventListener("change",()=>{state.flow=$("flow").value;state.subsystem="all";$("subsystem").value="all";state.selected=null;render();});
  $("edge-type").addEventListener("change",()=>{state.edgeType=$("edge-type").value;render();});
  $("layers").querySelectorAll("input").forEach(input=>input.addEventListener("change",()=>{input.checked?state.layers.add(input.value):state.layers.delete(input.value);render();}));
  $("external-only").addEventListener("change",()=>{state.external=$("external-only").checked;render();});
  $("integration-toggle").addEventListener("click",()=>{const expanded=$("integration-toggle").getAttribute("aria-expanded")!=="true";$("integration-toggle").setAttribute("aria-expanded",String(expanded));$("integrations").hidden=!expanded;});
  $("zoom-in").addEventListener("click",()=>zoomBy(1.25));$("zoom-out").addEventListener("click",()=>zoomBy(1/1.25));$("fit").addEventListener("click",resetCamera);
  $("reset").addEventListener("click",()=>{Object.assign(state,{view:"overview",subsystem:"all",flow:"all",edgeType:"all",search:"",external:false,selected:null});state.layers=new Set(graph.groups.map(group=>group.id));$("search").value="";$("subsystem").value="all";$("flow").value="all";$("edge-type").value="all";$("external-only").checked=false;$("layers").querySelectorAll("input").forEach(input=>input.checked=true);$("integrations").hidden=true;$("integration-toggle").setAttribute("aria-expanded","false");render();});
  $("map").addEventListener("wheel",event=>{event.preventDefault();zoomBy(event.deltaY<0?1.1:1/1.1);},{passive:false});
  $("map").addEventListener("pointerdown",event=>{if(event.target.closest("[data-node]"))return;pan={x:event.clientX,y:event.clientY,panX:state.panX,panY:state.panY};$("map").setPointerCapture(event.pointerId);});
  $("map").addEventListener("pointermove",event=>{if(!pan)return;const rect=$("map").getBoundingClientRect();const scale=Math.max(base.width/rect.width,base.height/rect.height)/state.zoom;state.panX=pan.panX-(event.clientX-pan.x)*scale;state.panY=pan.panY-(event.clientY-pan.y)*scale;updateCamera();});
  $("map").addEventListener("pointerup",()=>pan=null);$("map").addEventListener("pointercancel",()=>pan=null);
  $("map").addEventListener("keydown",event=>{if(event.target.closest("[data-node]"))return;const amount=70/state.zoom;const key=event.key;if(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","=","-","Home"].includes(key))event.preventDefault();if(key==="ArrowLeft")state.panX-=amount;if(key==="ArrowRight")state.panX+=amount;if(key==="ArrowUp")state.panY-=amount;if(key==="ArrowDown")state.panY+=amount;if(key==="+"||key==="=")zoomBy(1.25);if(key==="-")zoomBy(1/1.25);if(key==="Home")resetCamera();updateCamera();});
  const initial=new URLSearchParams(window.location.search);if(byGroup.has(initial.get("subsystem"))){state.subsystem=initial.get("subsystem");$("subsystem").value=state.subsystem;}if(graph.flows.some(flow=>flow.id===initial.get("flow"))){state.flow=initial.get("flow");state.view="journeys";$("flow").value=state.flow;}if(byId.has(initial.get("node")))state.selected=initial.get("node");
  render();
})();

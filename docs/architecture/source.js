"use strict";
(() => {
  const path=new URLSearchParams(window.location.search).get("path");
  const snapshot=Object.prototype.hasOwnProperty.call(window.CHROMATIC_SOURCES,path)?window.CHROMATIC_SOURCES[path]:null;
  if(!snapshot){document.getElementById("source-error").textContent="This path is not in the generated source snapshot. Regenerate the explorer after changing source references.";document.getElementById("raw-source").removeAttribute("href");return;}
  document.title=`${path} | ChromaTensor source`;
  document.getElementById("source-title").textContent=path;
  document.getElementById("source-hash").textContent=`SHA-256 ${snapshot.sha256}`;
  document.getElementById("raw-source").href="../../"+path;
  const target=document.getElementById("source-lines"),fragment=document.createDocumentFragment();
  snapshot.text.split(/\r?\n/).forEach((line,index)=>{const row=document.createElement("li"),anchor=document.createElement("a"),code=document.createElement("code");row.id=`L${index+1}`;anchor.href=`#${row.id}`;anchor.textContent=String(index+1);anchor.setAttribute("aria-label",`Line ${index+1}`);code.textContent=line;row.append(anchor,code);fragment.append(row);});
  target.append(fragment);
  const requested=document.getElementById(window.location.hash.slice(1));if(requested)requested.scrollIntoView({block:"center"});
})();

'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="app-token"]').content;
const selected = new Set();
const rows = new Map();
let state = {jobs:[]};
let toastTimer;
let downloadingZip = false;
let uploadChain = Promise.resolve();
const supported = /\.(jpe?g|png|webp)$/i;
function notice(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 7000);
}
async function api(path, body, isForm=false) {
  const response = await fetch(path, {method:body === undefined ? 'GET':'POST', headers:body === undefined ? {} :
    isForm ? {'X-App-Token':token} : {'X-App-Token':token,'Content-Type':'application/json'},
    body:body === undefined ? undefined : isForm ? body : JSON.stringify(body)});
  const data = await response.json();
  if(!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function options() {
  const compress = $('compress').checked;
  return {scale:Number($('scale').value), quality:$('quality').value, dpi:$('dpi').checked, compress,
    compression:$('compression').value, target_mb:compress && $('target').value ? Number($('target').value) : null,
    format:$('format').value, base_name:$('naming').value === 'custom' ? $('base').value.trim() : '', output_dir:$('output-dir').value.trim()};
}
function updateSettings() {
  $('compression-options').hidden = !$('compress').checked;
  $('base-label').hidden = $('naming').value !== 'custom';
  try {localStorage.setItem('upscaler-settings', JSON.stringify(options()));} catch (_) {}
}
function fillSettings(o) {
  for(const id of ['scale','quality','format','compression']) if(o[id] !== undefined) $(id).value = o[id];
  for(const id of ['dpi','compress']) if(o[id] !== undefined) $(id).checked = Boolean(o[id]);
  $('target').value = o.target_mb || '';
  $('base').value = o.base_name || ''; $('naming').value = o.base_name ? 'custom':'original';
  $('output-dir').value = o.output_dir || ''; updateSettings();
}
try {const stored=JSON.parse(localStorage.getItem('upscaler-settings')); if(stored) fillSettings(stored);} catch (_) {}
for(const input of document.querySelectorAll('.settings input,.settings select')) input.addEventListener('change', updateSettings);
$('preset').addEventListener('change', () => {
  const preset = $('preset').value;
  if(preset === 'etsy') fillSettings({scale:2, quality:'high', dpi:true, compress:true, compression:'balanced', target_mb:2, format:'original'});
  if(preset === 'max') fillSettings({scale:4, quality:'high', dpi:false, compress:false, compression:'balanced', target_mb:null, format:'original'});
});
$('apply-settings').onclick = () => runControl('apply');
$('add-images').onclick = event => {event.stopPropagation();$('files').click();};
$('add-folder').onclick = event => {event.stopPropagation();$('folder').click();};
for(const id of ['files','folder']) $(id).onchange = event => {enqueueUploads(Array.from(event.target.files));event.target.value='';};
$('drop-zone').addEventListener('keydown', event => {if(event.target === $('drop-zone') && ['Enter',' '].includes(event.key)){event.preventDefault();$('files').click();}});
for(const type of ['dragenter','dragover']) $('drop-zone').addEventListener(type,event=>{event.preventDefault();$('drop-zone').classList.add('drag');});
$('drop-zone').addEventListener('dragleave',()=> $('drop-zone').classList.remove('drag'));
async function entryFiles(entry) {
  if(entry.isFile) return [await new Promise((resolve,reject)=>entry.file(resolve,reject))];
  const reader=entry.createReader(); const files=[];
  while(true){const batch=await new Promise((resolve,reject)=>reader.readEntries(resolve,reject));if(!batch.length)break;
    for(const child of batch) files.push(...await entryFiles(child));}
  return files;
}
$('drop-zone').addEventListener('drop',async event=>{
  event.preventDefault();$('drop-zone').classList.remove('drag');
  // Obtain entry handles before the browser invalidates the drag data store.
  const entries=Array.from(event.dataTransfer.items||[]).map(item=>item.webkitGetAsEntry?.()).filter(Boolean);
  const fallback=Array.from(event.dataTransfer.files);
  try {const files=[]; if(entries.length){for(const entry of entries)files.push(...await entryFiles(entry));}else files.push(...fallback);enqueueUploads(files);}
  catch(error){notice(error.message);}
});
function enqueueUploads(files) {
  const config=options(); const valid=files.filter(file=>supported.test(file.name));
  if(valid.length !== files.length) notice(`${files.length-valid.length} file non-image dilewati.`);
  if(!valid.length)return;
  uploadChain=uploadChain.then(async()=>{
    let failed=0;
    for(let i=0;i<valid.length;i++) {
      $('upload-progress').textContent=`Adding ${i+1} / ${valid.length}: ${valid[i].name}`;
      const form=new FormData();form.append('images',valid[i]);form.append('options',JSON.stringify(config));
      try {await api('/api/upload',form,true);} catch(error){failed++;notice(error.message);}
      if(i%5 === 0) await refresh();
    }
    $('upload-progress').textContent=`${valid.length-failed} images added${failed?`, ${failed} failed`:''}.`;
    await refresh();
  }).catch(error=>notice(error.message));
}
async function runControl(action) {
  if(['cancel','remove','apply'].includes(action) && !selected.size){notice('Pilih gambar di antrean terlebih dahulu.');return;}
  if(action === 'clear' && !confirm('Clear queue? Completed output files will be kept.'))return;
  try {await api('/api/control',{action,ids:[...selected],options:action === 'apply'?options():undefined});await refresh();}
  catch(error){notice(error.message);}
}
for(const button of document.querySelectorAll('[data-control]')) button.onclick=()=>runControl(button.dataset.control);
$('select-all').onchange=event=>{selected.clear();if(event.target.checked)for(const job of state.jobs)selected.add(job.id);syncSelection();};
function syncSelection() {
  for(const [id, row] of rows) row.element.querySelector('input').checked=selected.has(id);
  $('select-all').checked=state.jobs.length>0 && selected.size===state.jobs.length;
  $('select-all').indeterminate=selected.size>0 && selected.size<state.jobs.length;
}
function el(tag,text='',className=''){const node=document.createElement(tag);if(text)node.textContent=text;if(className)node.className=className;return node;}
function size(bytes){return bytes>=1048576?`${(bytes/1048576).toFixed(2)} MB`:`${(bytes/1024).toFixed(1)} KB`;}
function renderRow(job) {
  const tr=el('tr');tr.dataset.id=job.id;
  const checkCell=el('td'),check=el('input');check.type='checkbox';check.setAttribute('aria-label',`Select ${job.filename}`);check.checked=selected.has(job.id);
  check.onchange=()=>{check.checked?selected.add(job.id):selected.delete(job.id);syncSelection();};checkCell.append(check);tr.append(checkCell);
  const previewCell=el('td');if(job.width){const image=el('img');image.src=`/api/thumbnail/${job.id}`;image.alt=job.filename;image.className='preview';image.loading='lazy';previewCell.append(image);}tr.append(previewCell);
  const name=el('td',job.filename,'filename');if(job.output_name)name.append(el('span',job.output_name,'sub'));tr.append(name);
  const resolution=el('td',`${job.width} × ${job.height}`);resolution.append(el('span',`→ ${job.output_width} × ${job.output_height}`,'sub'));tr.append(resolution);
  const scale=el('td',`${job.options.scale}×`);scale.append(el('span',job.options.quality === 'superfast' ? 'Super Cepat' : job.options.quality,'sub'));tr.append(scale);
  const status=el('td');status.append(el('span',`${job.status} · ${job.progress}%`,`status ${job.status}`));const bar=el('progress');bar.max=100;bar.value=job.progress;bar.setAttribute('aria-label',`${job.filename} progress`);status.append(bar);
  if(job.error || job.warning)status.append(el('div',job.error || job.warning,'message'));tr.append(status);
  const bytes=el('td',size(job.original_size));bytes.append(el('span',job.output_size?`→ ${size(job.output_size)}`:'→ —','sub'));tr.append(bytes);
  const cell=el('td'),actions=el('div','','actions');
  function button(label,action){const btn=el('button',label);btn.dataset.action=action;btn.dataset.id=job.id;actions.append(btn);}
  if(job.status==='Completed'){
    const download=el('a','Download');download.href=`/api/download/${job.id}`;actions.append(download);
    button('Open File','open');button('Open Folder','folder');button('Delete','delete');
  }else if(['Failed','Cancelled'].includes(job.status))button('Retry','retry');
  cell.append(actions);tr.append(cell);return tr;
}
$('queue').addEventListener('click',async event=>{
  const button=event.target.closest('button[data-action]');if(!button)return;
  const {action,id}=button.dataset;
  try{
    if(action==='delete'){if(!confirm('Delete this completed output file from disk?'))return;await api(`/api/delete/${id}`,{});}
    else if(action==='retry')await api('/api/control',{action:'retry',ids:[id]});
    else await api(`/api/open/${id}`,{folder:action==='folder'});
    await refresh();
  }catch(error){notice(error.message);}
});
$('zip').onclick=async()=>{try{await api('/api/zip',{});downloadingZip=true;await refresh();}catch(error){notice(error.message);}};
async function refresh() {
  state=await api('/api/state');const live=new Set(state.jobs.map(j=>j.id));
  for(const [id,row] of rows){if(!live.has(id)){row.element.remove();rows.delete(id);selected.delete(id);}}
  for(const job of state.jobs){const signature=JSON.stringify(job);const current=rows.get(job.id);
    if(!current || current.signature !== signature){const element=renderRow(job);if(current)current.element.replaceWith(element);else $('queue').append(element);rows.set(job.id,{element,signature});}}
  syncSelection();$('empty').hidden=state.total>0;$('queue-count').textContent=`${state.total} assets`;
  $('overall-count').textContent=`${state.completed} / ${state.total} Completed`;
  $('overall').value=state.total?state.jobs.reduce((sum,j)=>sum+(['Failed','Cancelled','Completed'].includes(j.status)?100:j.progress),0)/state.total:0;
  $('run-status').textContent=state.paused?'Paused':state.running?'Processing':'Ready';
  $('pause').textContent=state.paused?'Resume':'Pause';$('pause').dataset.control=state.paused?'resume':'pause';
  $('device').textContent=state.device==='cuda'?'Device: NVIDIA GPU · CUDA':state.device==='mps'?'Device: Apple GPU':state.device==='cpu'?'Device: CPU':`Device: AUTO · AI model ${state.model_progress}%`;
  const zip=state.zip;$('zip').disabled=zip.status==='Processing';
  $('zip-progress').textContent=zip.status==='Processing'?`Preparing final assets ZIP… ${zip.progress}%`:zip.status==='Failed'?zip.error:'';
  if(downloadingZip && zip.status==='Completed'){downloadingZip=false;const a=el('a');a.href='/api/zip/download';a.download='ETSY_UPSCALED_IMAGES.zip';document.body.append(a);a.click();a.remove();notice('ZIP ready. Only final image assets included.');}
  if(zip.status==='Failed')downloadingZip=false;
}
async function poll(){try{await refresh();}catch(error){$('run-status').textContent='Disconnected · keep the app running';}finally{setTimeout(poll,1000);}}
poll();

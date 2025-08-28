
(function(){
  const mathToggle = document.getElementById('mathToggle');
  const specToggle = document.getElementById('specToggle');
  const setMath = (on)=>{
    document.querySelectorAll('.math-block').forEach(el=>{ el.style.display = on ? 'block' : 'none'; });
    document.querySelectorAll('.math-inline').forEach(el=>{ el.style.display = on ? 'inline' : 'none'; });
    if(on && window.MathJax && window.MathJax.typeset){ window.MathJax.typeset(); }
    localStorage.setItem('dream_math_on', on ? '1' : '0');
  };
  const setSpec = (on)=>{
    document.querySelectorAll('.speculative').forEach(el=> el.style.display = on ? '' : 'none');
    localStorage.setItem('dream_spec_on', on ? '1' : '0');
  };
  const initMath = localStorage.getItem('dream_math_on') === '1';
  const initSpec = localStorage.getItem('dream_spec_on') === '1';
  if(mathToggle){ mathToggle.checked = initMath; setMath(initMath); }
  if(specToggle){ specToggle.checked = initSpec; setSpec(initSpec); }
  if(mathToggle){ mathToggle.addEventListener('change', e=> setMath(e.target.checked)); }
  if(specToggle){ specToggle.addEventListener('change', e=> setSpec(e.target.checked)); }
  document.querySelectorAll('.cloud').forEach(cloud=>{
    cloud.addEventListener('click', ()=> cloud.classList.toggle('open'));
  });
})();

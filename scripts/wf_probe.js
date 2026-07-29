export const meta = {
  name: 'model-probe',
  description: 'Probe whether claude-opus-5 and claude-fable-5 are routable as subagent models',
  phases: [{ title: 'probe' }],
}
const tries = [
  { label: 'opus5', model: 'claude-opus-5' },
  { label: 'fable5', model: 'claude-fable-5' },
  { label: 'opus-alias', model: 'opus' },
  { label: 'fable-alias', model: 'fable' },
];
const R = await parallel(tries.map(function (t) {
  return function () {
    return agent('Reply with ONLY your exact model id string, nothing else.', { label: t.label, phase: 'probe', model: t.model })
      .then(function (r) { return { asked: t.model, label: t.label, reply: String(r).slice(0, 120) }; })
      .catch(function (e) { return { asked: t.model, label: t.label, error: String(e).slice(0, 200) }; });
  };
}));
log('probe results: ' + JSON.stringify(R));
return R;

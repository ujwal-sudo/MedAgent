import urllib.request, json
r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7860/reset', data=b'', headers={'Content-Type': 'application/json'}))
out1 = json.loads(r.read())
print('STEP C PASS:', list(out1.keys()))

r2=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7860/step', data=b'{"action_type": "classify_urgency", "payload": {"urgency": "routine"}}', headers={'Content-Type': 'application/json'}))
out2 = json.loads(r2.read())
print('STEP D PASS: terminated=', out2['terminated'], 'observation=', out2['observation'].get('message', ''))

r3=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7860/step', data=b'{"action_type": "ask_symptom", "payload": {"symptom_id": "fever"}}', headers={'Content-Type': 'application/json'}))
out3 = json.loads(r3.read())
print('STEP E PASS: terminated=', out3['terminated'], 'reward=', out3['reward'], 'observation=', out3['observation'].get('message', ''))

r4=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:7860/state'))
out4 = json.loads(r4.read())
print('STEP F PASS: keys=', list(out4.keys()))

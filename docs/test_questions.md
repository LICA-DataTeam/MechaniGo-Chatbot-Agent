Perfect — that’s a great setup for testing your AI agent’s coverage and robustness 👌

Here’s a structured list of **possible customer questions** grouped by category. These will help you simulate realistic user interactions for your test.

---

## 🚗 **PMS (Preventive and Periodic Maintenance Service) Questions**

### 🔹 General PMS Inquiries

1. What is a PMS or preventive maintenance service?
2. How often should I bring my car for PMS?
3. What does PMS include?
4. Do you follow the manufacturer’s recommended PMS schedule?
5. How long does a PMS usually take?
6. Can I bring any car brand for PMS?
7. Do I need to go to the dealership for PMS, or can I have it done at MechaniGo?
8. Is PMS mandatory to keep my warranty valid?
9. What are the signs that my car needs PMS soon?
10. Can I skip a PMS schedule if my car seems fine?

### 🔹 Oil Change–Specific

1. How often should I change my engine oil?
2. What type of oil do you use? (mineral, synthetic, semi-synthetic?)
3. How can I tell if my car needs an oil change?
4. What happens if I don’t change my oil regularly?
5. Can I bring my own oil and filter?
6. How much does an oil change cost?
7. Do you replace the oil filter during oil change?
8. How long does an oil change take?
9. What’s included in your oil change package?
10. Can I book an oil change on weekends? *(good to test your “can’t handle booking” limitation)*

### 🔹 PMS Pricing and Coverage

1. How much is the PMS for a Toyota Vios 2020?
2. Do PMS rates differ by car model or engine type?
3. Is labor included in your PMS pricing?
4. Do you offer PMS packages or promos?
5. Can I get an estimate before availing PMS?

---

## 🔧 **Car Diagnosis & Troubleshooting Questions**

### 🔹 Engine Problems

1. My car is hard to start in the morning — what could be the problem?
2. There’s white smoke coming from my exhaust — what does that mean?
3. My car suddenly loses power while driving, what should I check?
4. Why is my “Check Engine” light on?
5. My engine is overheating — what should I do?
6. There’s a knocking noise coming from the engine, what could it be?
7. Why does my car vibrate when idling?
8. My fuel consumption suddenly increased — what’s wrong?

### 🔹 Electrical Issues

1. My headlights are flickering — what causes that?
2. My battery keeps draining — what should I check?
3. The car won’t start but the lights work — what could be wrong?
4. My alternator might be failing, how can I confirm that?
5. Why do my dashboard lights stay on after turning off the engine?

### 🔹 Air Conditioning & Comfort

1. My aircon isn’t getting cold — what could be the cause?
2. There’s a foul smell when I turn on the AC — how do I fix that?
3. My car’s AC blows cold air sometimes and warm air other times — why?
4. Why is there water dripping inside my car from the AC?

### 🔹 Brakes & Handling

1. My brakes squeak when I stop — is that normal?
2. The brake pedal feels soft — what could cause that?
3. My steering wheel vibrates when braking — what’s wrong?
4. The car pulls to one side when driving — why?
5. Why is my steering heavy?

### 🔹 Transmission & Driving

1. My car jerks when shifting gears — what could be wrong?
2. There’s a burning smell when driving uphill — should I worry?
3. Why does my automatic transmission shift roughly?
4. My clutch feels loose or slipping — what does that mean?

### 🔹 Noises & Vibrations

1. I hear a squealing sound when I start the car — what’s causing that?
2. There’s a clicking noise when I turn — what could it be?
3. Why does my car shake when accelerating?
4. I hear a thumping noise from under the car — what should I check?

---

## ⚠️ **Test Scenario Prompts (Edge Cases / Limitation Tests)**

These are useful to see if your agent handles limitations gracefully.

1. Can I book my PMS appointment now? *(should trigger “booking not available”)*
2. Can I give my details so you can remind me of my next oil change? *(should trigger “cannot extract user info”)*
3. Can you text me the price quote? *(same limitation)*
4. Can you send a mechanic to my house?
5. Can you help me schedule my PMS next week?

---

Would you like me to generate **sample test dialogues** next (like user–agent conversation samples) for each category to help you simulate real interactions and evaluate responses?

# Basic Symptoms

*Agent should acknowledge + ask ONE narrowing question, no tool call yet*

1. May kakaibang tunog yung sasakyan ko pag umaandar.

2. Parang mahina na yung hatak ng kotse ko.

3. Hindi masyadong lumalamig yung aircon.

4. May konting kalampag sa harap pag dumadaan sa lubak.

5. Mabilis maubos yung gas kahit normal lang driving ko.

# Clear Diagnosis Flow (Non-Critical)

*Agent should narrow down cause before suggesting next steps*

1. Tuwing umaga, hirap mag-start yung sasakyan ko pero okay na after.

2. May delay bago pumasok yung gear sa automatic transmission.

3. Pag naka-idel, nanginginig yung makina.

4. May konting usok pero nawawala rin pag mainit na makina.

5. Biglang bumaba yung fuel efficiency nitong mga nakaraang linggo.

# Safety-Critical Issues

*Agent SHOULD recognize risk; should call `mechanic_tool`*

1. Umiinit yung makina kahit short drive lang.

2. Parang malambot yung preno, lumulubog pag tinatapakan.

3. May amoy sunog pag naka-on yung aircon.

4. Biglang kumakabig yung manibela habang tumatakbo.

5. Safe po ba idrive kung may check engine light.

# Multi-Symptom / Complex Diagnosis

*Tool call likely, but only after minimal clarification*

1. May check engine light tapos mahina yung hatak at mausok.

2. May tunog sa ilalim tapos nanginginig pag humihinto.

3. Biglang namamatay makina pag traffic tapos hirap mag-start ulit.

4. May tagas sa ilalim tapos parang mabilis uminit makina.

5. May delay sa arangkada tapos mataas RPM.

# General Knowledge

*Agent should explain simply, no serice push or tool call*

1. Para san ba yung engine oil?

2. ANo pinagkaiba ng CVT at automatic?

3. Normal ba umiinit ang makina pag traffic?

4. Bakit mas malakas sa gas pag city driving?

5. Ano ginagawa ng radiator?

# Brand/Model/Year-specific

*Should trigger internal knowledge`*

1. Ano common issues ng Toyota Vios 2018?

2. Anong oil dapat ng Honda City 2020?

3. Gaano kadalas PMS ng Mitsubishi Montero? - Uses `faq_tool` most of the time

4. May recall ba sa Ford Everest 2018?
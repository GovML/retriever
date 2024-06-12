# Load model directly
from transformers import pipeline

question_answerer = pipeline("question-answering", model="AnonymousSub/specter-bert-model_squad2.0")
c = '''
Precise measurements of the timings in an exoplanet transit light curve is crucial for several reasons.
Mid-transit time is one such timing which should be determined with high precision in order to tell
whether it increases at a constant rate with the orbital period or the rate of change is variable for some
reason. In the presence of an additional but an unseen planet in the system, timing of the mid-point in a
transiting exoplanet’s light curve will fluctuate because of the gravitational perturbation of the additional
third body (Ballard et al. 2011), and to a greater extent, the light time effect (LiTE) due to the orbital
motion of the observed system around the center of mass with the unseen one (Holman & Murray
2005). This makes the detection of the unseen body possible from the Transit Timing Variations (TTVs) of
the observed transiting exoplanet. In fact, several non-transiting planets have been detected so far
thanks to TTVs they caused (Dawson et al. 2012, Saad-Olivera et al. 2017 and references therein) and the
validity of the technique have been confirmed for multi-body systems with more than one transiting
bodies (for Kepler-47 system see Vanderburg et al. 2017). These observations help theorists in their work
toward understanding the formation of planetary systems. Tidal interaction between a close-in planet
and its host star also causes an orbital period change, which is secular naturally, due to angular
momentum considerations.
'''
print(question_answerer(question='What can affect measuring the timing of exoplanet transit light curve?', context=c))
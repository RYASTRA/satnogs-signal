# satnogs-signal
Observation vetting / signal detection at scale. 
The network generates a firehose of observations, and a large fraction are empty — no signal, just noise. 
Deciding "did this observation actually catch the bird?" is still largely manual/heuristic. 
A model that looks at the waterfall image (and/or the IQ) and classifies signal-vs-noise — even just to triage — would improve data quality across the whole network. 
This is the one I'd bet on. It's a clean CV/ML problem, training data is effectively infinite and free, and it can live as a standalone service that consumes the API and posts results back. 
Prior GSoC efforts have nibbled at this but it's never been solved.

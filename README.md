Steps:

- Make a config file, samples are in ```config```
- Make data, e.g. with ```make_data_Leslie.py```
- Scale the data with ```scale_data.py```
- Train autoencoder and latent dynamics with ```train.py```
- Compute Morse graph with ```morse_graph.py```

Other files? 
There is a separate file for making the data for an arctangent map: ```make_data_arctan.py```
I wrote ```determine_domain.py``` to determine the domain for the 4D Leslie map.
I was doing PCA in the notebook ```Leslie_PCA.ipynb```.

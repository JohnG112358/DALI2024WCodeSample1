# DALI 2024W Application - Machine Learning Track
### John Guerrerio

This repository contains my code for code sample 1.  It contains the primary preprocessing task
for the LitCovid NER task (for more information, see the code description submitted with my application).  Given a file
in PubTator format (a file format used to represent a collection of scientific titles/abstracts and the named entities within them), this code tokenizes and preprocesses the text, generates IO and BIO labels for each document, 
and writes the results to a json that can be read by the transformers library.

## PubTator

Pubtator is a file format that represents a collection of titles and abstracts from scientific papers, and the named entities within them.  Given a single title, abstract, PMID (numerical identifier for a biomedical paper)
and named entities within the title/abstract, PubTator would format that information like so (note we ignore entity type for this task, as we are only doing named entity recognition):

PMID|t|Every scientific paper needs a title\
PMID|a|Every scientific paper also needs an abstract\
PMID<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization
PMID<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization
PMID<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization\
\n

Multiple papers are stacked on top of each other like so:

PMID1|t|The first paper's title\
PMID1|a|First paper's abstract\
PMID1<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization\
PMID1<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization\
PMID1<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization\
\n\
PMID2|t|Another one!\
PMID2|a|It has an abstract too\
PMID2<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization\
PMID2<code>&nbsp;</code>start_character<code>&nbsp;</code>end_character<code>&nbsp;</code>entity_text<code>&nbsp;</code>entity type<code>&nbsp;</code>entity normalization\
\n\

The below is two documents from the LitCovid test set to give a better idea of the format:

33499905|t|Israel's rapid rollout of vaccinations for COVID-19.\
33499905|a|As of the end of 2020, the State of Israel, with a population of 9.3 million,...\
33499905	2220	2226	Pfizer	Vaccine Funder	Pfizer\
33499905	2227	2235	BioNTech	Vaccine Funder	BioNTech

34953513|t|Strong humoral immune responses against SARS-CoV-2 Spike...\
34953513|a|The standard regimen of the BNT162b2 mRNA vaccine for SARS-CoV-2 includes two doses...\
34953513	63	71	BNT162b2	Vaccine	BioNTech;Pfizer\
34953513	156	164	BNT162b2	Vaccine	BioNTech;Pfizer\
34953513	530	538	BNT162b2	Vaccine	BioNTech;Pfizer

## Files
- preprocessing.py: The python code that preprocesses a PubTator file
- LitCovid_Val_Test.PubTator.txt: Sample PubTator file to preprocess
- LitCovid_combined.json: Output of preprocessing.py on LitCovid_Val_Test.PubTator.txt

## Notes 
- This code was written in 2022, as such it may not work with modern Python/libraries.  Please use Python 3.10 and the libraries specified in requirements.txt - I can't guarantee it will run otherwise
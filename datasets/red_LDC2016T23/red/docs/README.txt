  	DEFT Richer Event Description (RED) Annotation Corpus


1. Introduction
This is a set of 95 documents annotated with the Richer Event 
Description (RED) annotation scheme. The annotation is based on
source documents under data/source/, which were treated
as plain text. 

Annotation and source documents are divided into three partitions; 20 documents 
of newswire summarization documents in the "proxy" folder, 20 documents of discussion 
forum and newswire annotations used in the original RED pilot annotations, and 55 documents 
annotated by a range of DEFT annotation formats, whose other annotations were released in 
LDC2015E40.

2. Data format
The annotation data is under /data/annotation/, which contains
the annotation documents (extension ".RED-Relation.gold.completed.xml"). 

      
Documentations for this annotation are included in ./docs/
  red.dtd
  redguidelines.html
  splits.txt 

The redguidelines.html document is a snapshot of the official guidelines, 
hosted at https://github.com/timjogorman/RicherEventDescription. 
splits.txt is a randomly selected list of files for train/dev/test splits of 
the data, in case any users wish to evaluate upon RED.  
The red.dtd is a basic dtd for the xml format.

The Annotations are in folders within /data/annotation/ , with  
".RED-Relation.gold.completed.xml" extensions, and have the following 
annotation distribution
 
Events	        8731
Entities        10319
TIMEX3	        893
SECTION/DOCTIME 232

IDENT chains    2049
Partial IDENT   515, 947 383  (part/whole, set/member, and bridging)
TLINKs          4209
Reporting       596  

3. Annotation Pipeline

Documents were annotated using the "Anafora" tool 
(https://github.com/weitechen/anafora), in two passes.

The first pass involved the annotation of events, entities, TIMEX3 and 
section time elements across the entire document, while marking features 
upon each markable.  This is detailed in sections 3-8 in the included 
guidelines (/docs/red-guidelines.html).  These markables were 
double-annotated and then adjudicated.

The second pass involved the marking of relations between those 
adjudicated entities, as detailed in sections 9-14 in the guidelines.  
These relations were adjudicated, and quality control scripts were run 
to double-check any implausible feature combinations. 

For questions, issues, or the latest guidelines, consult the github page for the
guidelines, at https://github.com/timjogorman/RicherEventDescription .

4. Using the Data

4.1 XML file details

<entity> elements each have an "id" field.  These are used as the 
identifiers in <relation> annotations. <span> fields represent the 
location of the mention, using the direct utf-8 offsets into the text 
files located in /data/source (counting all xml characters).

5. Contact Information

Tim O'Gorman <ogormant@colorado.edu>
Martha Palmer <martha.palmer@colorado.edu>

-------------------

README Update Log
  Created: Tim O'Gorman, December 15, 2014
  Updated: April 1, 2015
  Updated: April 13, 2015, Zhiyi Song
  Updated: August 2, 2016, Tim O'Gorman

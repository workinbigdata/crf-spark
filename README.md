#CRF-Spark#
CRF on top of Spark

###Start###
- Build
-- sbt assembly


###Defining a template###

Features are extracted from the input using a template. That template contains a set of rules that are applied to each line in the input (note that EOF should not appear in the file, it's here to help your shell).

``` cat > chunking.template <

U00:%x[-2,0]
U01:%x[-1,0]
U02:%x[0,0]
U03:%x[1,   -
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++0]
U04:%x[2,0]
U05:%x[-1,0]/%x[0,0]
U06:%x[0,0]/%x[1,0]

U10:%x[-2,1]
U11:%x[-1,1]
U12:%x[0,1]
U13:%x[1,1]
U14:%x[2,1]
U15:%x[-2,1]/%x[-1,1]
U16:%x[-1,1]/%x[0,1]
U17:%x[0,1]/%x[1,1]
U18:%x[1,1]/%x[2,1]

U20:%x[-2,1]/%x[-1,1]/%x[0,1]
U21:%x[-1,1]/%x[0,1]/%x[1,1]
U22:%x[0,1]/%x[1,1]/%x[2,1]

B
B01%x[0,0] B02%x[0,1] B03%x[0,0]/%x[0,1] B04%x[-1,0]/%x[0,0] B05%x[-1,1]/%x[0,1]

EOF ```

If a rule begins by U, for unigram, it generates a feature for the current label, it it begins with a B, for bigram, it's for the joint label represented by the current and the previous labels. Then, a rule contains free text to identify the feature (two rules that cannot be identified by the free text are treated as if comming from the same bag) and one or more reference to the input %x[i,j] where i is the relative line number and j is the column number starting from zero. So, %x[0,0] is the current word and %x[-1,1] is the previous part-of-speech tag.

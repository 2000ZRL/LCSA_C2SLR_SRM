#!/bin/bash 
 
if [ -z "$2" ];then

echo "evaluateWER.sh <hypothesis-CTM-file> <dev | test>"
exit 0
fi

hypothesisCTM=$1
partition=$2
tmp_prefix=$3
workdir=evaluation_relaxation/

# apply some simplifications to the recognition
# remove __LEFTHAND__ and __EPENTHESIS__ and __EMOTION__ from ctm
# remove all words starting and ending with "__", 
# remove all -PLUSPLUS suffixes
# remove all cl- prefix
# remove all loc- prefix
# remove RAUM at the end (eg. NORDRAUM -> NORD)
# remove repetitions
cat ${workdir}${hypothesisCTM} | grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" | grep -v -e ' __[^_ ]*__$'| sed -e 's, loc-\([^ ]*\)$, \1,g' -e 's,-PLUSPLUS$,,g' -e 's, cl-\([^ ]*\)$, \1,g' -e 's,\b\([A-Z][A-Z]*\)RAUM$,\1,g' -e 's,\s*$,,'| awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' | awk 'BEGIN{prec=="";precID=""}{if (($NF!=prec)||($1!=precID)){print $0}precID=$1; prec=$NF}'  > ${workdir}$tmp_prefix.tmp.ctm

#make sure empty recognition results get filled with [EMPTY] tags - so that the alignment can work out on all data.
cat ${workdir}$tmp_prefix.tmp.ctm | sed -e 's,\s*$,,'   | awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' |sort -k1,1 -k3,3 > ${workdir}$tmp_prefix.tmp2.ctm

#in reference:
# remove __LEFTHAND__ and __EPENTHESIS__ and __EMOTION__
# remove all words starting and ending with "__", 
# remove all -PLUSPLUS suffixes
# remove all cl- prefix
# remove all loc- prefix
# remove RAUM at the end (eg. NORDRAUM -> NORD)
# join WIE AUSSEHEN to WIE-AUSSEHEN
# add spelling letters to compounds (A S -> A+S)
# remove repetitions
cat ${workdir}PHOENIX-2014-T-groundtruth-$partition.stm | sort  -k1,1 | sed -e 's/__LEFTHAND__ //g' -e 's/ __LEFTHAND__//g' -e 's/ __EPENTHESIS__//g' -e 's/__EPENTHESIS__ //g' -e 's/ __EMOTION__//g' -e 's/__EMOTION__ //g'| sed -e 's,\b__[^_ ]*__\b,,g' -e 's,\bloc-\([^ ]*\)\b,\1,g' -e 's,\bcl-\([^ ]*\)\b,\1,g' -e 's,\b\([^ ]*\)-PLUSPLUS\b,\1,g' -e 's,\b\([A-Z][A-Z]*\)RAUM\b,\1,g' -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g'  -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;' > ${workdir}$tmp_prefix.tmp.stm

#add missing entries, so that sclite can generate alignment
python ./${workdir}mergectmstm_T.py ${workdir}$tmp_prefix.tmp2.ctm ${workdir}$tmp_prefix.tmp.stm  

mv ${workdir}$tmp_prefix.tmp2.ctm ${workdir}$tmp_prefix.tmp.out.${hypothesisCTM}

#make sure NIST sclite toolbox is installed and on path. Available at ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.0-20091110-0958.tar.bz2
sclite  -h ${workdir}$tmp_prefix.tmp.out.$hypothesisCTM ctm -r ${workdir}$tmp_prefix.tmp.stm stm -f 0 -o sgml sum rsum pra    
sclite  -h ${workdir}$tmp_prefix.tmp.out.$hypothesisCTM ctm -r ${workdir}$tmp_prefix.tmp.stm stm -f 0 -o dtl stdout |grep Error

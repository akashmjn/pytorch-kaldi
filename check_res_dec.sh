#!/bin/bash
<<<<<<< HEAD
=======
for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | ./best_wer.sh; done
for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/*score_*/*.sys 2>/dev/null | ./best_wer.sh; done
exit 0


>>>>>>> master

# Compiles best WERs from decodings generated in passed dirs (exp dir containing decode_*)
for d in $@; do for ddecode in $d/decode_*; do echo $ddecode & grep Sum $ddecode/*scor*/*ys | ./best_wer.sh; done; done
exit 0


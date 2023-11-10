
###########################################################################
####### Training proposed methods in a joint manner
# example:
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --epochs 1000 \
    --workers 0 \
    --gpu 0 \
    --cr 64 \
    --L 5 \
    --mode Joint \
    --root /home/results/ \
    --scheduler const

####### For training proposed methods in a seperate manner
# Training FBNet:
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --epochs 1000 \
    --workers 0 \
    --gpu 0 \
    --cr 64 \
    --L 5 \
    --mode FB \
    --root /home/results/ \
    --scheduler const

# Training RENet:
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --epochs 1000 \
    --workers 0 \
    --gpu 0 \
    --L 5 \
    --mode RE \
    --root /home/results/ \
    --scheduler cosine
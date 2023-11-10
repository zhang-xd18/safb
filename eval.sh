
##################################################################
####### Evaluating the whole feedback performance 
### Evaluating the performance of Joint training:
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --workers 0 \
    --cpu \
    --cr 64 \
    --L 5 \
    --mode Joint \
    --root /home/results/ \
    --evaluate \
    --pretrained /home/Checkpoints/Joint-cr64-L5.pth

### Evaluating the performance of Separate training:
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --workers 0 \
    --cpu \
    --cr 64 \
    --L 5 \
    --mode Joint \
    --root /home/results/ \
    --evaluate \
    --pretrained /home/Checkpoints/FB-cr64-L5.pth \
    --pretrained2 /home/Checkpoints/RE-L5.pth

####### Evaluating the performance of FBNet
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --workers 0 \
    --cpu \
    --cr 64 \
    --L 5 \
    --mode FB \
    --root /home/results/ \
    --evaluate \
    --pretrained /home/Checkpoints/FB-cr64-L5.pth

####### Evaluating the performance of RENet
python /home/safb/main.py \
    --data-dir /home/SV/ \
    -b 200 \
    --workers 0 \
    --cpu \
    --L 5 \
    --mode RE \
    --root /home/results/ \
    --evaluate \
    --pretrained /home/Checkpoints/RE-L5.pth
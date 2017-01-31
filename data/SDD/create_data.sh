root_dir=$CAFFE_ROOT

cd $root_dir

redo=true
data_root_dir="$DATASETS/SDD"
dataset_name="SDD"
mapfile="$root_dir/data/$dataset_name/labelmap_SDD.prototxt"
anno_type="detection"
label_type="xml"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if $redo
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in train val test
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-type=$label_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset"_small".txt $data_root_dir/$db/$dataset_name"_small_"$subset"_"$db examples/$dataset_name 2>&1 | tee $root_dir/data/$dataset_name/$subset.log
done

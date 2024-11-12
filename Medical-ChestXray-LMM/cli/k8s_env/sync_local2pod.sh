from_path=/data2/checkpoints/MASAM-LITS/08012024/08012024_021901/model
dest_path=/media/hieu/6ac7d369-b609-4b09-97b0-27ed881b25f9/duong/checkpoints/MASAM-LITS/08012024/08012024_021901/model
pod_name=hieu-tms-data-transfer-pvc--1-r4pkh

kubectl cp ${pod_name}:${from_path} ${dest_path} 

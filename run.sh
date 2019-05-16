GROUP_NAME=cs449g13
APP_NAME=svm-spark
NB_EXECUTOR=4
REPO=tvaucher
TAG=latest

while getopts ":w:n:t:b" opt; do
  case $opt in
    w) NB_EXECUTOR="$OPTARG";; # number workers
    n) APP_NAME="$OPTARG";; # Name of the app
    t) TAG="$OPTARG";; # tag
    b) BUILD=1 ;; # Should we build the image or not
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

IMAGE=$REPO/$APP_NAME:$TAG
if [ -n "$BUILD" ] ; then
    bash docker-image-tool.sh -r $REPO -a $APP_NAME -t $TAG build
    bash docker-image-tool.sh -r $REPO -a $APP_NAME -t $TAG push
    sleep 1
fi;

kubectl delete pod $APP_NAME
spark-submit \
    --master k8s://https://10.90.36.16:6443 \
    --deploy-mode cluster \
    --name $APP_NAME \
    --conf spark.executor.instances=$NB_EXECUTOR \
    --conf spark.kubernetes.namespace=$GROUP_NAME \
    --conf spark.kubernetes.driver.pod.name=$APP_NAME \
	  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.volume1.options.claimName=$GROUP_NAME-scratch \
	  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.volume1.options.claimName=$GROUP_NAME-scratch \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.volume1.mount.path=/data \
	  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.volume1.mount.path=/data \
    --conf spark.kubernetes.container.image=$IMAGE \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.pyspark.pythonVersion="3" \
    local:///opt/spark/work-dir/hogwild.py
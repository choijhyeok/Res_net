USAGE="find-ecr-image — Check ECR for existing docker image

    Usage:
      ./find-ecr-image <repository-name> <image-tag>
    Example:
      ./find-ecr-image.sh foo/bar mytag
      ./find-ecr-image.sh -p public/repo mytag

    Options:
      <repository-name>   ECR repository name
      <image-tag>         ECR image tag 
      -h                  Show this message
      -p / --pubic        Public Repository (optional)
  "
  help() {
    echo "$USAGE"
  }

if [[ $# -lt 2 ]] || [[ "$1" == "-h" ]]; then
    help
    exit 1
fi

if [[ "$3" == "-p" ]] || [[ "$3" == "--public" ]]; then
    # public repository
    IMAGE_META="$( aws ecr-public describe-images --repository-name=$1 --image-ids=imageTag=$2 2> /dev/null )"
else
    # private repository
    IMAGE_META="$( aws ecr describe-images --repository-name=$1 --image-ids=imageTag=$2 2> /dev/null )"
fi

if [[ $? == 0 ]]; then
    IMAGE_TAGS="$( echo ${IMAGE_META} | jq '.imageDetails[0].imageTags[0]' -r )"
    echo "$1:$2 found"
else
    echo "$1:$2 not found"
    exit 1
fi
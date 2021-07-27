#!/bin/bash

#LOW="9001"
#HIGH="9999"
#export PORT=$((RANDOM * ($HIGH-$LOW+1) / 32768 + LOW))
#export PORT1=$(echo "'"$PORT"'")
#echo $PORT1

echo "-----------LOG------------"
echo -e '\t' " NEEDED VARIABLES"
echo "-----------LOG------------"
REGISTRY_USERNAME=""
REGISTRY_PASSWORD=""
REGISTRY_ORGANIZATION=""
IMAGE=""
declare -a tags_array=( "1.0.1" "1.0.0")



echo "-----------LOG------------"
echo -e '\t' "CEREATE LOGIN PAYLOAD"
echo "-----------LOG------------"
registry_login_payload()
{
cat << EOF
{ "username": "$REGISTRY_USERNAME","password": "$REGISTRY_PASSWORD"}
EOF
}


echo "-----------LOG------------"
echo -e '\t' "IMAGE TAGS" $(registry_login_payload)
echo "-----------LOG------------"

echo "-----------LOG------------"
echo -e '\t' "LOGIN TO THE REGISTRY_ORGANIZATION" $(registry_login_payload)
echo "-----------LOG------------"
TOKEN=`curl -s -H "Content-Type: application/json" -X POST -d "$(registry_login_payload)" "https://hub.docker.com/v2/users/login/" | jq -r .token`

echo "-----------LOG------------"
echo -e '\t' "TOKEN" ${TOKEN}
echo "-----------LOG------------"

echo "-----------LOG------------"
echo -e '\t' "ITERATE IMAGE TAGS"
echo "-----------LOG------------"
for image_tag in "${tags_array[@]}"
do
  echo "-----------LOG------------"
  echo -e '\t' "DELETE" $image_tag
  echo "-----------LOG------------"
  echo "-----------LOG------------"
  echo -e '\t' "GOING TO DELETE THIS" "https://hub.docker.com/v2/repositories/${REGISTRY_ORGANIZATION}/${IMAGE}/tags/${image_tag}/"
  echo "-----------LOG------------"
  delete_response=$(curl "https://hub.docker.com/v2/repositories/${REGISTRY_ORGANIZATION}/${IMAGE}/tags/${image_tag}/" -X DELETE -H "Authorization: JWT ${TOKEN}")
  echo "-----------LOG------------"
  echo -e '\t' "DELETE COMPLETED" $delete_response
  echo "-----------LOG------------"
done


echo



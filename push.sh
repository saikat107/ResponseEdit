echo 'Enter file name(s)'
read line
git add $line
git commit 
echo 'enter branch name:'
read line
git push origin $line  

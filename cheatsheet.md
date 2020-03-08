1- click on + icon and pick worksapce directory

2- add any local files using this command or click on + icon near the file name ('$ git add .' for all)

$ git add x y z

3- commit using this command or using the source control section

$ git commit -m "message"

4- now remote add respository using the following command or using command pallet with >Git: Add remote

$ git remote add origin https://github.com/url.git 

5- check the remote repo with this command

$ git remote -v

6- git pull first to get all info about the repo (incluing the branches) with 

$ git pull

7- git pull specific branch

$ git pull origin branchname

8- commit once more using this command or using the source control section (because after pulling the local file was deleted while merging!)

$ git commit -m "message"

9- now click on specific branch in the menu opened from clicking on the bottom left blue source control icon

10- now push the initally commited local files to the branch or click on the refrest icon in the bottom left corner

$ git push -u origin branchname

11- can push anytime

12- add with $git add remove with $git rm


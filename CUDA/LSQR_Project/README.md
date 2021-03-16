# PMPP_Project

## Compile Project
cd <Project_Directory>/build <br />
make <br />
./lsqr "<directory_to_file_of_MATRIX>" "<directory_to_file_of_Vector>" <lambda_value> <br />

## Command Upload-Folder
scp -r <folder_directory> <account>@gccg201.igd.fraunhofer.de:~/<directory_PATH>

## Command Upload-File
scp <file_in_directory> <account>@gccg201.igd.fraunhofer.de:~/<directory_PATH>

## Command Download-Folder
scp -r <account_name>@gccg201.igd.fraunhofer.de:~/<directory_PATH> <folder_directory> 

## Command Download-File
scp <account_name>@gccg201.igd.fraunhofer.de:~/<directory_PATH> <file_names> <folder_directory>

## ATTENTION 
MAKE SURE THAT YOU HAVE ALL .cc FILES IN THE MAKEFILE
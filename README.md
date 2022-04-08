## 1. Load images and metadata 

<details open>
<summary></summary>

<br>

Place image information and batch metadata in dataclasses to move through pipeline and append additional information. Each image will contain batch_id connecting it to it's upload and batch metadata.

</details>

<br>

## 2. Move to database

<details>
<summary></summary>

Make the initial transfer of image to the database. Physical images will not be moved into the database, only pointers to image paths will be moved. 

</details>
<br>

## 3. Detect plants

<details>
<summary></summary>


</details>
<br>

## 4. Convert detections from local to global coordinates

<details>
<summary></summary>


</details>
<br>


## 5. Move coordinates to database and save to local storage

<details>
<summary></summary>


</details>
<br>


## 6. Begin Annotation processing

<details>
<summary></summary>

Use local detection results saved as json files to segment vegetation

</details>
<br>



## 7. Train model

<details>
<summary></summary>


</details>
<br>

## 8. Train model

<details>
<summary></summary>


</details>
# DSS
Work repository "Metaheuristic algorithms applied to the concrete retaining walls problem "

## Links

[PostgresSQL database with results](https://drive.google.com/file/d/145GviBm9pH8PZBxuyZ_q3dktx5jz-4ad/view?usp=share_link)


[Graphics](https://drive.google.com/drive/folders/1A8VD6yyUZ8cHaatS_qXFmYHa-HOuz3TM?usp=share_link)


## Instructions

#### 1) Create PostgreSQL Database with the structure of Database/bd-RW-17-01-23-VSyQLSA.sql

#### 2) Update ".env" with the credentials corresponding to the created database.

#### 3) Populate database with the experiments to be performed in "configure.py".
```
python configure.py
```

#### 4) Running experiments
```
python main.py
```

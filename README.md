# NE-WNA
## Requirements
Please install pakeages by 
```javascript 
pip install -r requirements.txt
```
## Usage Example
Cora
```javascript 
python main.py --dataset cora --runs 100 --epochs 600 --batch_size 2200 --dropout 0 --hidden 300 --hidden_z 300 --early_stopping 20 --lr 0.007 --weight_decay 0.0004 --alpha 2 --beta 3 --tau 0.5 --order 4
```
CiteSeer
```javascript 
python main.py --dataset citeseer --epochs 500 --batch_size 3000 --lr 0.01 --weight_decay 0.0005 --dropout 0.8 --hidden 1500 --hidden_z 500 --early_stopping 10 --alpha 1 --beta 2 --tau 0.5 --order 4
```
PubMed
```javascript 
python main.py --dataset pubmed --runs 100 --epochs 400 --batch_size 2500 --lr 0.01 --weight_decay 0.0005 --dropout 0.2 --hidden 400 --hidden_z 400 --early_stopping 10 --alpha 10 --beta 1 --tau 0.5 --order 4
```
Amazon Computers
```javascript 
python main.py --dataset computers --runs 100 --epochs 300 --lr 0.005 --weight_decay 0.0005 --dropout 0.4 --hidden 400 --hidden_z 300 --early_stopping 10 --alpha 30 --beta 3 --tau 4 --order 6
```
Amazon Photo
```javascript 
python main.py --dataset photo --runs 100 --epochs 200 --lr 0.005 --weight_decay 0.0005 --dropout 0.5 --hidden 200 --hidden_z 200 --early_stopping 10 --alpha 25 --beta 3 --tau 4 --order 5
```
Coauthor CS
```javascript 
python main.py --dataset cs --runs 10 --epochs 100 --lr 0.01 --weight_decay 0.005 --dropout 0.8 --hidden 2000 --hidden_z 600 --early_stopping 10 --alpha 10 --beta 1 --tau 1.2 --order 2
```
## Results
model	|Cora	|CiteSeer	|PubMed|Amazon Computers	|Amazon Photo	|Coauthor CS
------ | -----  |----------- |---|--- | -----  |----------- |
NE-WNA|	82.8% |	74.2%|	82.5%|84.7%|	93.2% |	92.5%|

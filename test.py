using namespace std;
typedef struct BSTode{
    BSTode *left;
    BSTode *right;
    int data;
}BSTNode;
BSTNode *root;
int predt = -10000;
int judgeBST(BSTNode *bt){
    int b1,b2;
    if(bt==NULL){
    return 1;
}
    else{
    b1 = judgeBST(bt->left);
    if(b1==0||predict>bt->data) return 0;
    predt = bt->data;
    b2 = judgeBST(bt->right);
    return b2;
}
}
void preOrder(BSTNode *p){
    if(p!==NULL){
    cout<<p->data <<" ";
    preOrder(p->left);
    preOrder(p->right);
}
}

int main(void){

}
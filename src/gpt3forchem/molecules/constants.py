FEATURES = [
    "[CH4]",
    "[CH3]C",
    "[CH2](C)C",
    "[CH](C)(C)C",
    "[C](C)(C)(C)C",
    "[CH3][N,O,P,S,F,Cl,Br,I]",
    "[CH2X4]([N,O,P,S,F,Cl,Br,I])[A;!#1]",
    "[CH1X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])[A;!#1]",
    "[CH0X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])([A;!#1])[A;!#1]",
    "[C]=[!C;A;!#1]",
    "[CH2]=C",
    "[CH1](=C)[A;!#1]",
    "[CH0](=C)([A;!#1])[A;!#1]",
    "[C](=C)=C",
    "[CX2]#[A;!#1]",
    "[CH3]c",
    "[CH3]a",
    "[CH2X4]a",
    "[CHX4]a",
    "[CH0X4]a",
    "[cH0]-[A;!C;!N;!O;!S;!F;!Cl;!Br;!I;!#1]",
    "[c][#9]",
    "[c][#17]",
    "[c][#35]",
    "[c][#53]",
    "[cH]",
    "[c](:a)(:a):a",
    "[c](:a)(:a)-a",
    "[c](:a)(:a)-C",
    "[c](:a)(:a)-N",
    "[c](:a)(:a)-O",
    "[c](:a)(:a)-S",
    "[c](:a)(:a)=[C,N,O]",
    "[C](=C)(a)[A;!#1]",
    "[C](=C)(c)a",
    "[CH1](=C)a",
    "[C]=c",
    "[CX4][A;!C;!N;!O;!P;!S;!F;!Cl;!Br;!I;!#1]",
    "[#6]",
    "[#1][#6,#1]",
    "[#1]O[CX4,c]",
    "[#1]O[!C;!N;!O;!S]",
    "[#1][!C;!N;!O]",
    "[#1][#7]",
    "[#1]O[#7]",
    "[#1]OC=[#6,#7,O,S]",
    "[#1]O[O,S]",
    "[#1]",
    "[NH2+0][A;!#1]",
    "[NH+0]([A;!#1])[A;!#1]",
    "[NH2+0]a",
    "[NH1+0]([!#1;A,a])a",
    "[NH+0]=[!#1;A,a]",
    "[N+0](=[!#1;A,a])[!#1;A,a]",
    "[N+0]([A;!#1])([A;!#1])[A;!#1]",
    "[N+0](a)([!#1;A,a])[A;!#1]",
    "[N+0](a)(a)a",
    "[N+0]#[A;!#1]",
    "[NH3,NH2,NH;+,+2,+3]",
    "[n+0]",
    "[n;+,+2,+3]",
    "[NH0;+,+2,+3]([A;!#1])([A;!#1])([A;!#1])[A;!#1]",
    "[NH0;+,+2,+3](=[A;!#1])([A;!#1])[!#1;A,a]",
    "[NH0;+,+2,+3](=[#6])=[#7]",
    "[N;+,+2,+3]#[A;!#1]",
    "[N;-,-2,-3]",
    "[N;+,+2,+3](=[N;-,-2,-3])=N",
    "[#7]",
    "[o]",
    "[OH,OH2]",
    "[O]([A;!#1])[A;!#1]",
    "[O](a)[!#1;A,a]",
    "[O]=[#7,#8]",
    "[OX1;-,-2,-3][#7]",
    "[OX1;-,-2,-2][#16]",
    "[O;-0]=[#16;-0]",
    "[O-]C(=O)",
    "[OX1;-,-2,-3][!#1;!N;!S]",
    "[O]=c",
    "[O]=[CH]C",
    "[O]=C(C)([A;!#1])",
    "[O]=[CH][N,O]",
    "[O]=[CH2]",
    "[O]=[CX2]=O",
    "[O]=[CH]c",
    "[O]=C([C,c])[a;!#1]",
    "[O]=C(c)[A;!#1]",
    "[O]=C([!#1;!#6])[!#1;!#6]",
    "[#8]",
    "[#9-0]",
    "[#17-0]",
    "[#35-0]",
    "[#53-0]",
    "[#9,#17,#35,#53;-]",
    "[#53;+,+2,+3]",
    "[+;#3,#11,#19,#37,#55]",
    "[#15]",
    "[S;-,-2,-3,-4,+1,+2,+3,+5,+6]",
    "[S-0]=[N,O,P,S]",
    "[S;A]",
    "[s;a]",
    "[#3,#11,#19,#37,#55]",
    "[#4,#12,#20,#38,#56]",
    "[#5,#13,#31,#49,#81]",
    "[#14,#32,#50,#82]",
    "[#33,#51,#83]",
    "[#34,#52,#84]",
    "[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30]",
    "[#39,#40,#41,#42,#43,#44,#45,#46,#47,#48]",
    "[#72,#73,#74,#75,#76,#77,#78,#79,#80]",
]


TEXT = ["smiles", "selfies", "iupac_name"]
TARGETS = ["logP"]
CAT_TARGETS = ["logP_binned"]

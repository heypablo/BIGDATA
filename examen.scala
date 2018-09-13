scala>def Par(n:Int):Boolean={
     | if(n%2==0){
     | return true
     | }else
     | return false
     | }

scala> def Parli(n:List[Int])={
      | for(i<-n)
      | println(Par(i))
      | }

scala> def sumli(i:List[Int])={
      | for(n <- i)
      | if(n==7)
      | c = 14 + c
      | else
      | c = c + n
      | println(s"$c")
      | }
scala> def equilibra(eq:List[Int]):Boolean={
      | if(eq.length%2==0)
      | return true
      | else
      | return false
      | }

scala> def Pali(word:String):Boolean ={
      | return (word == word.reverse)
      | }

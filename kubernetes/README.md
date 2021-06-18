1. kubernetes установлен в облаке https://mcs.mail.ru/containers/
````(base) imd@imd-desktop:~$ kubectl cluster-info
Kubernetes control plane is running at https://185.241.193.78:6443
CoreDNS is running at https://185.241.193.78:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
````

3. Ресурсы нужны для выбора хоста на котором запустить приложение. 
   Будет выбран тот хост который лучше всего удовлетворяет запросам.
   Лимиты нужны для ограничения ресурсов которые может потребить приложение, 
   при превышении лимитов приложение будет остановлено.
   
